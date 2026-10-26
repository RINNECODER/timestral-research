#!/usr/bin/env python3
"""Fine-tune Timestral on Coinbase BTC 1m (pre-cut only) with pinball loss, then compare zero-shot vs fine-tuned
on the late_half exactly like eval.py.

The training forward is TimestralForecaster.forecast itself with its @inference_mode stripped (`__wrapped__`), so
masking, causal patch loc/scale, asinh, the quantile head, denormalization and symmetric averaging are the
library's own code. Cut = median open_ts of markets.parquet (PLAN.md): every training/val window ends before it.

  python finetune.py --data data --out . --model mistralai/Timestral-28M --steps 3000
"""
import argparse, json, math, shutil, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file
from timestral import TimestralForecaster
from timestral.loading import resolve_checkpoint

try:
    import eval as ev  # local
except ImportError:
    import eval_mod as ev  # Kaggle dataset copy of eval.py

CTX, H = ev.CTX, 16
TAUS = torch.tensor(ev.LEVELS, dtype=torch.float32)
W_H = torch.ones(H); W_H[4:15] = 2.0  # steps 5..15 = the horizons eval uses (k=10,5,1 -> h=5,10,14) count double
W_H /= W_H.mean()
forecast_grad = TimestralForecaster.forecast.__wrapped__  # same function, no inference_mode


def load_series(data, cut):
    b = pd.read_parquet(data / "btc_1m_long.parquet").sort_values("ts")
    b = b[b.ts < cut]
    ts, x = b.ts.to_numpy(np.int64), b.close.to_numpy(np.float32)
    assert (np.diff(ts) == 60).all(), "expected a full 1-minute grid"
    return ts, x


def window_starts(x, lo, hi):
    """Starts i with x[i:i+CTX+H] inside [lo, hi), finite target + last close, <=20% NaN in the last 30 ctx mins."""
    nanc = np.concatenate([[0], np.cumsum(np.isnan(x))])
    i = np.arange(lo, hi - (CTX + H) + 1)
    ok = (nanc[i + CTX + H] - nanc[i + CTX] == 0) & ~np.isnan(x[i + CTX - 1]) & (nanc[i + CTX] - nanc[i + CTX - 30] <= 6)
    return i[ok]


def check_no_leak(ts, starts, cut):
    last = ts[starts + CTX + H - 1]
    assert len(starts) and (last < cut).all(), "a window touches ts >= cut"
    assert (ts[starts + CTX + H - 1] - ts[starts] == 60 * (CTX + H - 1)).all()


def windows(x, starts):
    w = x[starts[:, None] + np.arange(CTX + H)]
    return w[:, :CTX], w[:, CTX:]


def pinball(q, y, ctx):
    """Mean weighted pinball over 21 levels x 16 steps, in units of a random-walk sd (last*sig1m*sqrt(h))."""
    r = np.diff(np.log(ctx[:, -121:]), axis=1)
    sig = np.clip(np.nan_to_num(np.nanstd(r, axis=1)), 1e-5, None)
    s = torch.as_tensor(ctx[:, -1] * sig, device=q.device)[:, None] * torch.sqrt(torch.arange(1, H + 1, device=q.device))
    u = (torch.as_tensor(y, device=q.device)[..., None] - q.float()) / s[..., None]
    t = TAUS.to(q.device)
    return (torch.maximum(t * u, (t - 1) * u).mean(-1) * W_H.to(q.device)).mean()


def predict(f, ctx):
    return forecast_grad(f, ctx[:, None, :], H).quantiles[:, 0]  # (B, H, Q) raw price


def amp(dev):
    if dev != "cuda":
        return torch.autocast("cpu", enabled=False), None
    dt = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16  # T4 = fp16
    return torch.autocast("cuda", dtype=dt), torch.amp.GradScaler(enabled=dt == torch.float16)


def val_loss(f, x, starts, dev, bs):
    tot = 0.0
    with torch.no_grad():
        for i in range(0, len(starts), bs):
            c, y = windows(x, starts[i:i + bs])
            with amp(dev)[0]:
                tot += float(pinball(predict(f, c), y, c)) * len(c)
    return tot / len(starts)


def save(f, name, out):
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(resolve_checkpoint(name) / "config.json", out / "config.json")
    save_file({k: v.detach().float().cpu().contiguous() for k, v in f.model.state_dict().items()}, out / "model.safetensors")


def train(a, dev, cut):
    ts, x = load_series(a.data, cut)
    vstart = int(len(x) * 0.9)
    tr, va = window_starts(x, 0, vstart), window_starts(x, vstart, len(x))
    check_no_leak(ts, tr, cut); check_no_leak(ts, va, cut)
    assert ts[tr + CTX + H - 1].max() < ts[vstart]  # train windows end before val time
    rng = np.random.default_rng(0)
    va = np.sort(rng.choice(va, min(a.val_n, len(va)), replace=False))
    print(f"cut {cut} train windows {len(tr)} val {len(va)} (val from ts {ts[vstart]})", flush=True)

    f = TimestralForecaster.from_pretrained(a.model, device=dev)
    opt = torch.optim.AdamW(f.model.parameters(), lr=a.lr, weight_decay=0.01)
    warm = max(1, a.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps)))))
    ctx_amp, scaler = amp(dev)
    ckpt = a.out / f"ft_{a.tag}"
    best = v0 = val_loss(f, x, va, dev, a.bs * 2)
    log = [dict(step=0, val=v0)]; save(f, a.model, ckpt)
    print(f"step 0 val {v0:.4f}", flush=True)
    t0, run, skipped = time.time(), [], 0
    for step in range(1, a.steps + 1):
        c, y = windows(x, rng.choice(tr, a.bs))
        f.model.train()
        try:
            with ctx_amp:
                loss = pinball(predict(f, c), y, c)
        except ValueError:  # forecast() raises on NaN output (fp16 overflow); drop the batch
            skipped += 1; continue
        opt.zero_grad(set_to_none=True)
        if scaler: scaler.scale(loss).backward(); scaler.unscale_(opt)
        else: loss.backward()
        torch.nn.utils.clip_grad_norm_(f.model.parameters(), 1.0)
        if scaler: scaler.step(opt); scaler.update()
        else: opt.step()
        sched.step(); run.append(float(loss))
        last = step == a.steps or (time.time() - t0) / 60 > a.max_minutes
        if step % a.eval_every == 0 or last:
            v = val_loss(f, x, va, dev, a.bs * 2)
            log.append(dict(step=step, train=float(np.mean(run)), val=v)); run = []
            if v < best: best = v; save(f, a.model, ckpt)
            print(f"step {step} train {log[-1]['train']:.4f} val {v:.4f} best {best:.4f} "
                  f"{(time.time() - t0) / step:.2f}s/step", flush=True)
        if last: break
    return ckpt, dict(val_zero_shot=v0, val_best=best, steps_done=step, skipped=skipped, log=log)


def evaluate(a, ckpt, cut):
    ev.DATA = a.data
    df, ctxs = ev.build()
    late = (df.open_ts >= cut).to_numpy()  # PLAN cut; training data all < cut
    df, ctxs = df[late].reset_index(drop=True), ctxs[late]
    if a.eval_n:
        keep = np.sort(np.random.default_rng(0).choice(len(df), min(a.eval_n, len(df)), replace=False))
        df, ctxs = df.iloc[keep].reset_index(drop=True), ctxs[keep]
    hs, st = df.h.to_numpy(), df.strike.to_numpy()
    df["p_zs"] = ev.run_model(a.model, ctxs, hs, st)
    df["p_ft"] = ev.run_model(str(ckpt), ctxs, hs, st)  # reload through from_pretrained
    cols = ["mid", "p_rw", "p_zs", "p_ft"]
    res = {}
    for k in ev.OFFSETS:
        d = df[df.k == k]
        r = {c: ev.scores(d[c].to_numpy(), d.y.to_numpy()) for c in cols}
        for c in cols[1:]:
            r[c]["trade"] = {str(e): ev.trade(d[c], d.bid, d.ask, d.y, e) for e in ev.EDGES}
        res[f"k{k}_late_half"] = dict(n=len(d), **r)
        print(f"k{k} late n={len(d)} " + " ".join(f"{c} {r[c]['brier']:.4f}" for c in cols), flush=True)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    p.add_argument("--out", type=Path, default=Path("."))
    p.add_argument("--model", default="mistralai/Timestral-28M")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--val_n", type=int, default=1024)
    p.add_argument("--eval_n", type=int, default=0, help="subsample late_half rows (0 = all)")
    p.add_argument("--max_minutes", type=float, default=70)
    a = p.parse_args()
    a.tag = a.model.split("-")[-1]
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cut = int(pd.read_parquet(a.data / "markets.parquet").open_ts.median())
    ckpt, tr = train(a, dev, cut)
    res = evaluate(a, ckpt, cut)
    path = a.out / "results_finetune.json"
    allres = json.loads(path.read_text()) if path.exists() else {}
    allres[a.tag] = dict(model=a.model, cut=cut, args={k: str(v) for k, v in vars(a).items()}, train=tr, late_half=res,
                         note="late_half = open_ts >= cut (median markets.open_ts); 1 fine-tune config per model")
    path.write_text(json.dumps(allres, indent=1))
    print(f"wrote {path}", flush=True)
