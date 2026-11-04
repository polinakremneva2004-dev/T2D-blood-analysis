"""
run_all.py

Runs the whole pipeline in order. Each step can also be run on its own.

    python run_all.py              # steps 1 to 7
    python run_all.py --from 4     # restart from step 4 (after quantification)

Step 8 (comparison with the authors' GEO matrix) needs a file you download
by hand, so it is run separately; see 08_compare_geo_matrix.py.
"""

import argparse
import subprocess
import sys

STEPS = [
    (1, "01_download_subsample.py"),
    (2, "02_qc_trim.py"),
    (3, "03_quantify.py"),
    (4, "04_explore.py"),
    (5, "05_differential_expression.py"),
    (6, "06_cell_composition.py"),
    (7, "07_extension_repeated_measures.py"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="start", type=int, default=1)
args = ap.parse_args()

for n, script in STEPS:
    if n < args.start:
        continue
    print(f"\n{'=' * 70}\nSTEP {n}: {script}\n{'=' * 70}", flush=True)
    rc = subprocess.call([sys.executable, script])
    if rc != 0:
        sys.exit(f"Step {n} failed, stopping.")
