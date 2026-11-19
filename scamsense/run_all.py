"""Runs the whole pipeline: make data -> train -> discover scam families."""
import subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

for step in ("1_make_dataset.py", "2_train.py", "3_discover.py"):
    print(f"\n==================== {step} ====================")
    if subprocess.call([sys.executable, os.path.join(HERE, step)]) != 0:
        sys.exit(f"Step {step} failed - see the message above.")
print("\nDone! Now run:  python app.py")
