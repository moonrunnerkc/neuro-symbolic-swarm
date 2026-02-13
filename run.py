#!/usr/bin/env python3
# Author: Bradley R. Kinnard
"""Convenience launch script for neuro-symbolic-swarm."""

import os
import sys
from pathlib import Path

# prevent C-level heap corruption from FAISS + sentence-transformers
# fighting over OpenMP thread pools. must be set before any imports.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ensure project root is on the path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.main import run

if __name__ == "__main__":
    sys.exit(run())
