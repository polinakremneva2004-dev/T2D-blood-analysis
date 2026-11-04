"""
06_cell_composition.py

Step 6: does blood cell composition differ between groups, and how much of
the step 5 DE signal could it explain?

Approach 1 (default): marker-gene scores
  For each cell type, the mean z-score of its marker genes (config.py,
  CELL_MARKERS). NLR_proxy = neutrophil score minus mean lymphocyte
  (T, B, NK) score, a transcriptomic analogue of the neutrophil-to-
  lymphocyte ratio.
  Assumptions, state these in the report:
    * marker expression per cell is constant across people and conditions,
      so marker level tracks cell abundance (inflammation can violate this,
      e.g. activated neutrophils change transcription);
    * markers are specific to one circulating cell type;
    * scores are RELATIVE between samples, not absolute proportions.
  The script checks the first assumption indirectly: markers of the same
  cell type should correlate with each other (mean pairwise r). Low
  coherence means that score is unreliable.

Approach 2 (optional): reference-based deconvolution
  Non-negative least squares (as in CIBERSORT without the SVR step) against
  a published signature matrix you supply, e.g. LM22 (Newman et al. 2015),
  downloadable from the CIBERSORTx website after free registration:
      python 06_cell_composition.py --signature LM22.txt
  Extra assumptions: the reference profiles (from sorted cells, other
  people, partly microarray) match cells in these samples, and the
  signature covers all cell types present. LM22 omits platelets and
  red cell precursors, which whole blood contains.

Then:
  * test each score between groups (Welch t-test and Mann-Whitney, BH)
  * re-run DE adjusting for NLR_proxy: ~ group + NLR_proxy. Only ONE
    composition covariate, because with 12 samples each added term costs a
    residual degree of freedom (9 samples - 3 coefficients = 6 left).
  * compare unadjusted vs adjusted fold changes, and check whether genes
    that differ between groups are the same genes that track composition.

Outputs: results/tables/06_*.tsv, results/figures/06_*.png
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import optimize, stats

import config as C

GROUP_COLOURS = {"case": "#c0392b", "control": "#2c7fb8"}


def nnls_deconvolution(tpm: pd.DataFrame, ann: pd.DataFrame, sig_path: str) -> pd.DataFrame:
    sig = pd.read_csv(sig_path, sep="\t", index_col=0)
    by_name = tpm.groupby(ann.loc[tpm.index, "gene_name"]).sum()
    genes = sig.index.intersection(by_name.index)
    print(f"  signature genes found: {len(genes)} / {len(sig)}")
    S = sig.loc[genes].values
    props = {}
    for s in by_name.columns:
        coef, _ = optimize.nnls(S, by_name.loc[genes, s].values)
        props[s] = coef / coef.sum() if coef.sum() > 0 else coef
    return pd.DataFrame(props, index=sig.columns).T


def group_tests(scores: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    rows = []
    for ct in scores.columns:
        a, b = scores.loc[group == "case", ct], scores.loc[group == "control", ct]
        t, p_t = stats.ttest_ind(a, b, equal_var=False)
        _, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append({"cell_type": ct, "mean_case": a.mean(), "mean_control": b.mean(),
                     "diff_case_minus_control": a.mean() - b.mean(),
                     "welch_t": t, "p_welch": p_t, "p_mannwhitney": p_u})
    out = pd.DataFrame(rows).set_index("cell_type")
    out["padj_welch"] = C.bh(out["p_welch"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signature", help="optional signature matrix (genes x cell types, TSV)")
    args = ap.parse_args()

    C.ensure_dirs()
    samples = C.load_samples("primary")
    counts, ann = C.load_counts(samples)
    filt = C.filter_genes(counts)
    lognorm = C.log_norm(filt)

    # ---- Marker scores -----------------------------------------------------
    scores, coherence = C.marker_scores(lognorm, ann)
    coherence.round(3).to_csv(C.TABLES / "06_marker_coherence.tsv", sep="\t", index=False)
    print("Marker coherence (mean pairwise r of markers within each cell type):")
    print(coherence.round(2).to_string(index=False))
    weak = coherence.loc[coherence["mean_pairwise_r"] < 0.3, "cell_type"].tolist()
    if weak:
        print(f"  Low coherence, interpret with caution: {weak}")
    scores.round(4).to_csv(C.TABLES / "06_marker_scores.tsv", sep="\t")

    tests = group_tests(scores, samples["group"])
    tests.round(4).to_csv(C.TABLES / "06_composition_group_tests.tsv", sep="\t")
    print("\nComposition scores, case vs control:")
    print(tests[["diff_case_minus_control", "p_welch", "p_mannwhitney", "padj_welch"]].round(3).to_string())

    # ---- Optional reference deconvolution ----------------------------------
    if args.signature:
        tpm = pd.read_csv(C.COUNTS / "gene_tpm.tsv", sep="\t", index_col=0)[samples.index]
        props = nnls_deconvolution(tpm, ann, args.signature)
        props.round(4).to_csv(C.TABLES / "06_nnls_proportions.tsv", sep="\t")
        ptests = group_tests(props.loc[:, props.std() > 0], samples["group"])
        ptests.round(4).to_csv(C.TABLES / "06_nnls_group_tests.tsv", sep="\t")
        print("\nNNLS deconvolution, case vs control:")
        print(ptests[["mean_case", "mean_control", "p_welch", "padj_welch"]].round(3).to_string())

    # ---- DE with and without composition adjustment ------------------------
    meta = samples.copy()
    meta["nlr"] = scores["NLR_proxy"]
    print("\nRunning DE: ~ group (unadjusted) and ~ group + NLR_proxy (adjusted)")
    unadj = C.run_deseq(filt, meta, factors=["group"])
    adj = C.run_deseq(filt, meta, factors=["group", "nlr"], continuous=["nlr"])

    cmp_ = pd.DataFrame({
        "gene_name": ann.loc[unadj.index, "gene_name"],
        "baseMean": unadj["baseMean"],
        "log2FC_unadj": unadj["log2FoldChange"], "p_unadj": unadj["pvalue"], "padj_unadj": unadj["padj"],
        "log2FC_adj": adj["log2FoldChange"], "p_adj": adj["pvalue"], "padj_adj": adj["padj"],
    })
    # How strongly each gene tracks composition across samples
    cmp_["rho_with_NLR"] = lognorm.T.corrwith(scores["NLR_proxy"], method="spearman")
    cmp_ = cmp_.sort_values("p_unadj")
    cmp_.to_csv(C.TABLES / "06_de_unadjusted_vs_adjusted.tsv", sep="\t")

    ok = cmp_.dropna(subset=["log2FC_unadj", "log2FC_adj"])
    top = ok[ok["p_unadj"] < 0.01]
    kept = (top["p_adj"] < 0.01).sum()
    rho_fc, p_fc = stats.spearmanr(ok["log2FC_unadj"], ok["rho_with_NLR"])
    summary = {
        "genes padj<0.05 unadjusted": int((cmp_["padj_unadj"] < 0.05).sum()),
        "genes padj<0.05 adjusted": int((cmp_["padj_adj"] < 0.05).sum()),
        "genes p<0.01 unadjusted": len(top),
        "  of which still p<0.01 after adjustment": int(kept),
        "  of which |rho with NLR_proxy| > 0.6": int((top["rho_with_NLR"].abs() > 0.6).sum()),
        "correlation of log2FC unadj vs adj (all genes)": round(ok["log2FC_unadj"].corr(ok["log2FC_adj"]), 3),
        "median |log2FC| unadj, p<0.01 genes": round(top["log2FC_unadj"].abs().median(), 3) if len(top) else np.nan,
        "median |log2FC| adj, same genes": round(top["log2FC_adj"].abs().median(), 3) if len(top) else np.nan,
        "Spearman(gene log2FC, gene rho with NLR), all genes": f"{rho_fc:.3f} (p={p_fc:.2g})",
    }
    pd.Series(summary).to_csv(C.TABLES / "06_adjustment_summary.tsv", sep="\t", header=False)
    print("\nEffect of adjusting for composition:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nInterpretation guide: if case-control fold changes line up with how genes "
          "track NLR_proxy (positive Spearman above) and shrink after adjustment, the "
          "unadjusted signal is at least partly a shift in cell mix rather than "
          "within-cell regulation. Caveat: if NLR_proxy itself differs between groups, "
          "group and composition are partly collinear, so this Spearman is inflated "
          "by construction and adjustment can also remove real group signal that acts "
          "through composition. With n = 9 the two cannot be cleanly separated.")

    # ---- Figures -----------------------------------------------------------
    plot_cols = [c for c in ["neutrophil", "monocyte", "T_cell", "CD8_T", "B_cell", "NK",
                             "eosinophil", "platelet", "erythroid", "NLR_proxy"] if c in scores]
    long = scores[plot_cols].join(samples["group"]).melt(id_vars="group", var_name="cell type",
                                                          value_name="score")
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.boxplot(data=long, x="cell type", y="score", hue="group", palette=GROUP_COLOURS,
                ax=ax, showfliers=False)
    sns.stripplot(data=long, x="cell type", y="score", hue="group", palette=GROUP_COLOURS,
                  dodge=True, ax=ax, edgecolor="k", linewidth=0.5, legend=False)
    for i, ct in enumerate(plot_cols):
        ax.text(i, ax.get_ylim()[1] * 0.95, f"p={tests.loc[ct, 'p_welch']:.2f}",
                ha="center", fontsize=7)
    ax.set_title("Marker-based cell-type scores (relative, z-score units)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(C.FIGS / "06_composition_scores.png", dpi=200)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(ok["log2FC_unadj"], ok["log2FC_adj"], s=3, alpha=0.4, c="grey")
    axes[0].scatter(top["log2FC_unadj"], top["log2FC_adj"], s=8, c="#c0392b", label="p<0.01 unadjusted")
    lim = np.nanmax(np.abs(ok[["log2FC_unadj", "log2FC_adj"]].values))
    axes[0].plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    axes[0].set_xlabel("log2FC, ~ group")
    axes[0].set_ylabel("log2FC, ~ group + NLR_proxy")
    axes[0].legend(frameon=False)
    axes[0].set_title("Fold changes before and after adjustment")
    axes[1].scatter(ok["rho_with_NLR"], ok["log2FC_unadj"], s=3, alpha=0.4, c="grey")
    axes[1].set_xlabel("gene correlation with NLR_proxy (Spearman)")
    axes[1].set_ylabel("log2FC case vs control (unadjusted)")
    axes[1].set_title(f"Do DE genes track composition? rho={rho_fc:.2f}")
    fig.tight_layout()
    fig.savefig(C.FIGS / "06_adjustment.png", dpi=200)
    print(f"\nFigures written to {C.FIGS}")


if __name__ == "__main__":
    main()
