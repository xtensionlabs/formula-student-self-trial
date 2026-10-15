"""Make the packages importable for plain `pytest tests/` without sourcing the workspace."""
import glob
import os
import sys

for p in glob.glob(os.path.join(os.path.dirname(__file__), '..', 'src', '*')):
    sys.path.insert(0, p)
