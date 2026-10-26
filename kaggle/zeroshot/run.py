import glob, os, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/mistralai/timestral.git"], check=True)
data = os.path.dirname(glob.glob("/kaggle/input/**/markets.parquet", recursive=True)[0])
src = glob.glob("/kaggle/input/**/eval_mod.py", recursive=True)[0]
subprocess.run([sys.executable, src, data, "/kaggle/working"], check=True)
