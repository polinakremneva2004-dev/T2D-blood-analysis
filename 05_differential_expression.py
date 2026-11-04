"""
05_differential_expression.py

Step 5: differential expression, future T2D converters vs non-converters,
baseline samples that passed QC (4 case vs 5 control after excluding
three failed libraries; see config.EXCLUDE).

What this comparison contrasts: all 12 samples were drawn at BASELINE,
before any subject had transitioned. "case" therefore means "will develop
T2D before follow-up", not "has T2D". The test asks whether the blood
transcriptome of people who later convert already differs from that of
people who do not, i.e. a pre-diagnostic / risk-state contrast.

Model: PyDESeq2 (negative binomial GLM, Python port of DESeq2), design
~ group, reference level = control, Wald test, Benjamini-Hochberg FDR with
DESeq2 independent filtering and Cook's distance outlier handling.
Age and sex are not in the model: all samples are female and groups are
age-matched by design (age was checked against the PCs in step 4).

Also reported:
  * p-value histogram: under no true signal this is flat. A flat histogram
    is the honest evidence for "nothing here", beyond "0 genes at FDR 5%".
  * Approximate power: the smallest |log2 fold change| detectable with 80%
    power given the observed standard errors.

Outputs: results/tables/05_de_results.tsv, results/figures/05_*.png

Usage:
    python 05_differential_expression.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import config as C


def detectable_lfc(res: pd.DataFrame, n_tests: int) -> pd.DataFrame:
    """Minimum |log2FC| detectable with 80% power for a two-sided Wald test,
    MDE = (z_{1-alpha/2} + z_{0.8}) * SE, using the observed lfcSE.
    Two thresholds bracket what BH needs when there are few true positives:
    alpha = 0.05 / n_tests (Bonferroni-like) and alpha = 0.001 (lenient)."""
    z80 = stats.norm.ppf(0.8)
    rows = []
    r = res.dropna(subset=["lfcSE"]).copy()
    r["expr_quartile"] = pd.qcut(np.log10(r["baseMean"] + 1), 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    for alpha_label, alpha in [(f"0.05/{n_tests}", 0.05 / n_tests), ("0.001", 0.001)]:
        zc = stats.norm.ppf(1 - alpha / 2)
        for q, sub in r.groupby("expr_quartile", observed=True):
            se = sub["lfcSE"].median()
            mde = (zc + z80) * se
            rows.append({"alpha": alpha_label, "expression_quartile": q,
                         "median_lfcSE": se, "min_detectable_abs_log2FC": mde,
                         "min_detectable_fold_change": 2 ** mde})
    return pd.DataFrame(rows)


def main() -> None:
    C.ensure_dirs()
    samples = C.load_samples("primary")
    counts, ann = C.load_counts(samples)
    filt = C.filter_genes(counts)
    print(f"Testing {len(filt):,} genes, {sum(samples.group == 'case')} case vs "
          f"{sum(samples.group == 'control')} control (baseline samples)")

    res = C.run_deseq(filt, samples, factors=["group"])
    res = ann[["gene_name", "gene_type"]].join(res, how="right").sort_values("pvalue")
    res.to_csv(C.TABLES / "05_de_results.tsv", sep="\t")

    n_tested = int(res["padj"].notna().sum())
    summary = {
        "genes_in_model": len(res),
        "genes_after_independent_filtering": n_tested,
        "padj<0.05": int((res["padj"] < 0.05).sum()),
        "padj<0.10": int((res["padj"] < 0.10).sum()),
        "nominal p<0.001": int((res["pvalue"] < 0.001).sum()),
        "nominal p<0.01": int((res["pvalue"] < 0.01).sum()),
        "expected p<0.01 under null": round(0.01 * res["pvalue"].notna().sum(), 1),
    }
    pd.Series(summary).to_csv(C.TABLES / "05_de_summary.tsv", sep="\t", header=False)
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nTop 15 genes by p-value:")
    print(res.head(15)[["gene_name", "baseMean", "log2FoldChange", "lfcSE", "pvalue", "padj"]]
          .round(4).to_string())

    power = detectable_lfc(res, max(n_tested, 1))
    power.round(3).to_csv(C.TABLES / "05_power.tsv", sep="\t", index=False)
    print("\nApproximate minimum detectable effect (80% power):")
    print(power.round(2).to_string(index=False))

    # ---- Figures -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].hist(res["pvalue"].dropna(), bins=40, color="grey", edgecolor="white")
    axes[0].axhline(res["pvalue"].notna().sum() / 40, color="k", ls="--", lw=1,
                    label="expected if no signal")
    axes[0].set_xlabel("p-value")
    axes[0].set_ylabel("genes")
    axes[0].set_title("p-value distribution")
    axes[0].legend(frameon=False)

    sig = res["padj"] < C.FDR
    axes[1].scatter(res["log2FoldChange"], -np.log10(res["pvalue"]), s=4,
                    c=np.where(sig, "#c0392b", "grey"), alpha=0.6)
    for _, r in res.head(10).iterrows():
        axes[1].annotate(r["gene_name"], (r["log2FoldChange"], -np.log10(r["pvalue"])), fontsize=7)
    axes[1].set_xlabel("log2 fold change (case / control)")
    axes[1].set_ylabel("-log10 p")
    axes[1].set_title(f"Volcano ({int(sig.sum())} genes FDR<{C.FDR})")

    axes[2].scatter(np.log10(res["baseMean"] + 1), res["log2FoldChange"], s=4,
                    c=np.where(sig, "#c0392b", "grey"), alpha=0.6)
    axes[2].axhline(0, color="k", lw=0.8)
    axes[2].set_xlabel("log10 mean normalised count")
    axes[2].set_ylabel("log2 fold change")
    axes[2].set_title("MA plot")
    fig.tight_layout()
    fig.savefig(C.FIGS / "05_de_overview.png", dpi=200)
    print(f"\nResults: {C.TABLES / '05_de_results.tsv'}")


if __name__ == "__main__":
    main()
