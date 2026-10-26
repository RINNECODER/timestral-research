import glob, os, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/mistralai/timestral.git"], check=True)
data = os.path.dirname(glob.glob("/kaggle/input/**/btc_1m_long.parquet", recursive=True)[0])
src = glob.glob("/kaggle/input/**/finetune.py", recursive=True)[0]
# 28M first; 209M only runs on what's left (each capped by --max_minutes, kernel limit is far above ~2h total)
for model, steps, bs, lr in [("mistralai/Timestral-28M", 4000, 64, "1e-5"), ("mistralai/Timestral-209M", 3000, 32, "1e-5")]:
    subprocess.run([sys.executable, src, "--data", data, "--out", "/kaggle/working", "--model", model,
                    "--steps", str(steps), "--bs", str(bs), "--lr", lr, "--max_minutes", "50"], check=False)
