"""Import helper: lets scripts run without sourcing the workspace (adds src/* to sys.path)."""
import glob
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # small matrices + many worker processes
os.environ.setdefault("OMP_NUM_THREADS", "1")
for p in glob.glob(os.path.join(os.path.dirname(__file__), "..", "src", "*")):
    sys.path.insert(0, p)
