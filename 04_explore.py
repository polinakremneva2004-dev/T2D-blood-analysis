"""
04_explore.py

Step 4: expression-level QC and exploratory analysis of the 12 primary
(baseline) samples.

  1. Expression QC: genes detected, % reads on globin, mitochondrial and rRNA
     genes, and a sex check (XIST vs chrY genes; all samples should be female)
  2. Filter low-count genes and normalise (DESeq2 median-of-ratios, log2)
  3. Outlier check from sample-to-sample correlation
  4. PCA on the most variable genes
  5. Correlate each PC with candidate drivers: group, age, original sequencing
     depth, mapping rate, Q30, duplication, globin %, and cell-type marker
     scores. This answers "what is the dominant axis of variation?"

Outputs: results/tables/04_*.tsv and results/figures/04_*.png

Usage:
    python 04_explore.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA

import config as C

GROUP_COLOURS = {"case": "#c0392b", "control": "#2c7fb8"}


def expression_qc(counts: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
    total = counts.sum()
    name = ann["gene_name"]
    globin = counts[name.isin(C.GLOBIN_GENES)].sum()
    mito = counts[name.str.startswith("MT-", na=False)].sum()
    rrna = counts[ann["gene_type"].isin(["rRNA", "Mt_rRNA", "rRNA_pseudogene"])].sum()
    xist = counts[name == "XIST"].sum()
    ygenes = counts[name.isin(C.SEX_GENES_Y)].sum()
    top10 = counts.apply(lambda c: c.nlargest(10).sum())
    return pd.DataFrame({
        "assigned_reads": total,
        "genes_detected_ge10": (counts >= 10).sum(),
        "pct_reads_top10_genes": 100 * top10 / total,
        "pct_globin": 100 * globin / total,
        "pct_mito": 100 * mito / total,
        "pct_rRNA": 100 * rrna / total,
        "XIST_cpm": 1e6 * xist / total,
        "chrY_genes_cpm": 1e6 * ygenes / total,
    })


def correlation_outliers(lognorm: pd.DataFrame) -> pd.DataFrame:
    r = lognorm.corr(method="spearman")
    r = r.mask(np.eye(len(r), dtype=bool))      # ignore self-correlation
    mean_r = r.mean()
    med = mean_r.median()
    mad = (mean_r - med).abs().median() * 1.4826
    z = (mean_r - med) / mad if mad > 0 else mean_r * 0
    return pd.DataFrame({"mean_spearman_to_others": mean_r, "robust_z": z,
                         "flag_outlier": z < -3})


def pc_associations(pcs: pd.DataFrame, covars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    r_tab = pd.DataFrame(index=covars.columns, columns=pcs.columns, dtype=float)
    p_tab = r_tab.copy()
    for cv in covars.columns:
        x = covars[cv].astype(float)
        if x.nunique() < 2:
            continue
        for pc in pcs.columns:
            r, p = stats.pearsonr(x, pcs[pc])
            r_tab.loc[cv, pc], p_tab.loc[cv, pc] = r, p
    return r_tab, p_tab


def library_qc_figure(qc: pd.DataFrame) -> None:
    """One panel per diagnostic metric, all 16 libraries, excluded ones in
    black. Makes the case for exclusion visible at a glance."""
    panels = [("genes_detected_ge10", "genes with >= 10 reads"),
              ("pct_reads_top10_genes", "% reads in top 10 genes"),
              ("gc_pct", "GC content (%)"),
              ("duplication_pct", "duplication (%)"),
              ("pct_globin", "% reads on globin genes"),
              ("salmon_pct_mapped", "Salmon mapping rate (%)")]
    panels = [p for p in panels if p[0] in qc]
    lab = qc["subject"] + np.where(qc["timepoint"] == "followup", " (FU)", "")
    colours = np.where(qc["excluded"], "black",
                       qc["group"].map(GROUP_COLOURS).fillna("grey"))
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 1.9 * len(panels)), sharex=True)
    for ax, (col, title) in zip(np.atleast_1d(axes), panels):
        ax.bar(range(len(qc)), qc[col], color=colours)
        ax.set_ylabel(title, fontsize=8)
    ax.set_xticks(range(len(qc)))
    ax.set_xticklabels(lab, rotation=60, ha="right", fontsize=8)
    if "library_type" in qc:
        for i, lt in enumerate(qc["library_type"]):
            ax.annotate(lt, (i, 0), xytext=(0, -62), textcoords="offset points",
                        ha="center", fontsize=7, annotation_clip=False)
    np.atleast_1d(axes)[0].set_title(
        "Library QC, all 16 libraries (red case, blue control, black excluded; "
        "bottom labels: Salmon library type)", fontsize=9)
    fig.tight_layout()
    fig.savefig(C.FIGS / "04_library_qc.png", dpi=200)
    plt.close(fig)


def main() -> None:
    C.ensure_dirs()
    # ---- 1. Expression QC on ALL samples, incl. excluded ones -----------------
    all_samples = C.load_samples(include_excluded=True)
    all_counts, ann_all = C.load_counts(all_samples)
    eqc = expression_qc(all_counts, ann_all)
    print("Expression QC (all 16 libraries):\n", eqc.round(2).to_string())
    if (eqc["chrY_genes_cpm"] > 10).any():
        print("WARNING: chrY expression detected, check sex labels:",
              eqc.index[eqc["chrY_genes_cpm"] > 10].tolist())

    qc_all = all_samples.copy()
    for f in (C.QC / "read_qc_metrics.tsv", C.QC / "salmon_metrics.tsv"):
        if f.exists():
            extra = pd.read_csv(f, sep="\t", index_col=0)
            qc_all = qc_all.join(extra[[c for c in extra.columns if c not in qc_all.columns]])
    qc_all = qc_all.join(eqc)
    qc_all["exclusion_reason"] = pd.Series(C.EXCLUDE).reindex(qc_all.index).fillna("")
    qc_all.round(4).to_csv(C.TABLES / "04_library_qc_all.tsv", sep="\t")
    library_qc_figure(qc_all)
    if C.EXCLUDE:
        print("\nExcluded after QC:")
        for r, why in C.EXCLUDE.items():
            print(f"  {r}  {why}")

    # ---- Analysis set: primary samples that passed QC -----------------------
    samples = C.load_samples("primary")
    counts, ann = C.load_counts(samples)
    print(f"\nAnalysis set: {sum(samples.group == 'case')} case, "
          f"{sum(samples.group == 'control')} control")
    a, b = samples.loc[samples.group == "case", "age"], samples.loc[samples.group == "control", "age"]
    print(f"Age after exclusions: case {a.min()}-{a.max()} (mean {a.mean():.1f}), "
          f"control {b.min()}-{b.max()} (mean {b.mean():.1f}), "
          f"Welch p = {stats.ttest_ind(a, b, equal_var=False).pvalue:.2f}")
    qc = qc_all.loc[samples.index].copy()

    # ---- 2. Filter and normalise -------------------------------------------
    filt = C.filter_genes(counts)
    print(f"\nGenes: {len(counts):,} quantified, {len(filt):,} pass filter "
          f"(>= {C.MIN_COUNT} reads in >= {C.MIN_SAMPLES} samples)")
    sf = C.size_factors(filt)
    qc["size_factor"] = sf
    lognorm = C.log_norm(filt)

    # ---- 3. Outliers -------------------------------------------------------
    out = correlation_outliers(lognorm)
    qc = qc.join(out)
    print("\nSample correlation check:\n", out.round(3).to_string())
    qc.round(4).to_csv(C.TABLES / "04_sample_qc.tsv", sep="\t")

    # ---- 4. PCA ------------------------------------------------------------
    top = lognorm.var(axis=1).sort_values(ascending=False).index[: C.N_TOP_VAR_GENES]
    X = lognorm.loc[top].T
    X = X - X.mean()
    n_pc = min(6, X.shape[0] - 1)
    pca = PCA(n_components=n_pc).fit(X)
    pcs = pd.DataFrame(pca.transform(X), index=X.index,
                       columns=[f"PC{i+1}" for i in range(n_pc)])
    var = pd.Series(100 * pca.explained_variance_ratio_, index=pcs.columns)
    print("\nVariance explained (%):\n", var.round(1).to_string())

    # Genes driving each PC (top loadings), useful for interpreting axes
    load = pd.DataFrame(pca.components_.T, index=top, columns=pcs.columns)
    load_rows = []
    for pc in pcs.columns[:4]:
        for sign, idx in (("+", load[pc].nlargest(15).index), ("-", load[pc].nsmallest(15).index)):
            load_rows.append({"PC": pc, "direction": sign,
                              "genes": ",".join(ann.loc[idx, "gene_name"].fillna("NA"))})
    pd.DataFrame(load_rows).to_csv(C.TABLES / "04_pc_top_loadings.tsv", sep="\t", index=False)

    # ---- 5. What drives each PC? -------------------------------------------
    scores, _ = C.marker_scores(lognorm, ann)
    covars = pd.DataFrame({"group (case=1)": (samples["group"] == "case").astype(int),
                           "age": samples["age"],
                           "original depth": samples["reads_brief"]})
    for col, label in [("salmon_pct_mapped", "% mapped"), ("q30_before", "Q30"),
                       ("duplication_pct", "duplication"), ("pct_globin", "% globin"),
                       ("pct_mito", "% mito"), ("size_factor", "size factor")]:
        if col in qc:
            covars[label] = qc[col]
    for ct in ["neutrophil", "monocyte", "T_cell", "B_cell", "NK", "NLR_proxy"]:
        if ct in scores:
            covars[f"score: {ct}"] = scores[ct]

    r_tab, p_tab = pc_associations(pcs.iloc[:, :4], covars)
    r_tab.round(3).to_csv(C.TABLES / "04_pc_covariate_r.tsv", sep="\t")
    p_tab.to_csv(C.TABLES / "04_pc_covariate_p.tsv", sep="\t")
    print("\nPearson r between PCs and covariates:\n", r_tab.round(2).to_string())
    for pc in r_tab.columns[:2]:
        best = r_tab[pc].abs().idxmax()
        print(f"{pc} ({var[pc]:.1f}%) correlates most with '{best}' "
              f"(r={r_tab.loc[best, pc]:.2f}, p={p_tab.loc[best, pc]:.2g})")

    pd.concat([pcs, samples[["group", "subject", "age"]]], axis=1).to_csv(
        C.TABLES / "04_pca_scores.tsv", sep="\t")

    # ---- Figures -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(var.index, var.values, color="grey")
    axes[0].set_ylabel("% variance explained")
    axes[0].set_title("Scree plot")
    for ax, (a, b) in zip(axes[1:], [("PC1", "PC2"), ("PC3", "PC4")]):
        if b not in pcs:
            ax.axis("off")
            continue
        for g, col in GROUP_COLOURS.items():
            m = samples["group"] == g
            ax.scatter(pcs.loc[m, a], pcs.loc[m, b], c=col, s=60, label=g, edgecolor="k")
        for s, row in pcs.iterrows():
            ax.annotate(samples.loc[s, "subject"], (row[a], row[b]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(f"{a} ({var[a]:.1f}%)")
        ax.set_ylabel(f"{b} ({var[b]:.1f}%)")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(C.FIGS / "04_pca.png", dpi=200)

    fig, ax = plt.subplots(figsize=(6, 0.4 * len(r_tab) + 1.5))
    sns.heatmap(r_tab.astype(float), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                ax=ax, cbar_kws={"label": "Pearson r"})
    ax.set_title("What correlates with each principal component?")
    fig.tight_layout()
    fig.savefig(C.FIGS / "04_pc_covariates.png", dpi=200)

    r = lognorm.corr(method="spearman")
    lab = samples["subject"]
    g = sns.clustermap(r.rename(index=lab, columns=lab), cmap="viridis", figsize=(7, 7),
                       row_colors=samples["group"].map(GROUP_COLOURS).rename(index=lab))
    g.fig.suptitle("Sample-sample Spearman correlation", y=1.02)
    g.savefig(C.FIGS / "04_sample_correlation.png", dpi=200)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, col in zip(axes, ["pct_globin", "pct_mito", "genes_detected_ge10"]):
        sns.stripplot(data=qc.reset_index(), x="group", y=col, hue="group",
                      palette=GROUP_COLOURS, ax=ax, size=7, legend=False)
        ax.set_title(col)
    fig.tight_layout()
    fig.savefig(C.FIGS / "04_expression_qc.png", dpi=200)
    print(f"\nFigures written to {C.FIGS}")


if __name__ == "__main__":
    main()
