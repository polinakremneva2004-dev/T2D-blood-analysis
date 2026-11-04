"""
08_compare_geo_matrix.py

Optional sanity check: compare our gene-level quantification (from
subsampled FASTQ) with the authors' normalised matrix from the GEO
supplementary files of GSE184050. Not used for any analysis.

Download the supplementary matrix manually from the GEO page for
GSE184050, then:
    python 08_compare_geo_matrix.py path/to/author_matrix.txt.gz

The script tries to match author columns to our samples by GSM ID or run
ID. If the author columns use other sample names, give a two-column TSV
(author_column <tab> run) with --map.

Genes are matched by Ensembl ID (version suffix removed) or, failing that,
by gene symbol. Reported per sample: Spearman correlation between our
log2 normalised counts and the author values (log2 transformed if they
look unlogged). Expect high but imperfect agreement: different depth (2M
pairs vs full), quantifier, annotation release and normalisation.
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("author_matrix")
    ap.add_argument("--map", help="TSV: author_column, run")
    args = ap.parse_args()

    samples = C.load_samples(include_excluded=True)
    counts, ann = C.load_counts(samples)
    ours = C.log_norm(C.filter_genes(counts, min_samples=3))

    sep = "," if args.author_matrix.rstrip(".gz").endswith(".csv") else "\t"
    auth = pd.read_csv(args.author_matrix, sep=sep, index_col=0)
    auth = auth.select_dtypes("number")
    print(f"Author matrix: {auth.shape[0]:,} genes x {auth.shape[1]} columns")

    # ---- match samples -----------------------------------------------------
    if args.map:
        m = pd.read_csv(args.map, sep="\t", header=None, index_col=0)[1]
    else:
        gsm_to_run = samples.reset_index().set_index("gsm")["run"]
        m = {}
        for col in auth.columns:
            for key, run in list(gsm_to_run.items()) + [(r, r) for r in samples.index]:
                if key in str(col):
                    m[col] = run
        m = pd.Series(m, dtype=str)
    if m.empty:
        print("Could not match columns automatically. First columns:", list(auth.columns[:10]))
        print("Create a --map file (author_column <tab> run) and re-run.")
        return
    auth = auth[m.index].rename(columns=m)
    print(f"Matched {auth.shape[1]} samples")

    # ---- match genes -------------------------------------------------------
    idx = auth.index.astype(str).str.replace(r"\.\d+$", "", regex=True)
    our_ids = ours.index.str.replace(r"\.\d+$", "", regex=True)
    if idx.str.startswith("ENSG").mean() > 0.5:
        auth.index = idx
        ours.index = our_ids
    else:
        ours = ours.groupby(ann.loc[ours.index, "gene_name"].values).mean()
    shared = ours.index.intersection(auth.index)
    print(f"Shared genes: {len(shared):,}")
    a = auth.loc[shared]
    a = a.groupby(level=0).mean()
    if a.max().max() > 100:          # looks unlogged
        a = np.log2(a.clip(lower=0) + 1)
    o = ours.loc[a.index, a.columns]

    rho = pd.Series({s: o[s].corr(a[s], method="spearman") for s in a.columns}, name="spearman")
    rho.round(3).to_csv(C.TABLES / "08_geo_agreement.tsv", sep="\t")
    print("\nPer-sample Spearman rho, ours vs authors:\n", rho.round(3).to_string())

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    s = a.columns[0]
    ax.scatter(a[s], o[s], s=2, alpha=0.3)
    ax.set_xlabel("author value (log2)")
    ax.set_ylabel("our log2 normalised count")
    ax.set_title(f"{s}: rho = {rho[s]:.2f}")
    fig.tight_layout()
    fig.savefig(C.FIGS / "08_geo_agreement.png", dpi=200)


if __name__ == "__main__":
    main()
