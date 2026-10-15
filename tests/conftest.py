"""Make the packages importable for plain `pytest tests/` without sourcing the workspace."""
import os

# Small dense matrices: multi-threaded BLAS is ~10x slower here (thread oversubscription).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import glob
import sys

for p in glob.glob(os.path.join(os.path.dirname(__file__), '..', 'src', '*')):
    sys.path.insert(0, p)
