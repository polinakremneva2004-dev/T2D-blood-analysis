"""
07_extension_repeated_measures.py

Step 7 (extension): add the 4 follow-up samples (13 samples, 9 subjects
after QC exclusions; all four follow-up subjects have a usable baseline).

Why the model must change:
  * case_2, case_5, control_4 and control_8 now contribute two samples
    each. Two samples from one person are more alike than samples from two
    people, so treating all samples as independent (pseudoreplication) inflates
    the effective sample size and gives p-values that are too small.
  * Timepoint now varies, and the meaning of "case" changes with it: a case
    follow-up sample was drawn AFTER transition (likely diagnosed, possibly
    treated, e.g. metformin), whereas case baseline samples are
    pre-diagnosis. With only 2 follow-up samples per group, a
    group x timepoint interaction cannot be estimated with any precision.

Primary model, per gene, on log2 normalised expression:
    expr ~ group + timepoint,  with correlated errors within subject
  Same idea as limma's duplicateCorrelation (Smyth et al. 2005):
    1. fit a random-intercept mixed model per gene (statsmodels MixedLM,
       REML) only to estimate the within-subject correlation
       (ICC = subject variance / total variance);
    2. take ONE consensus correlation across genes (trimmed mean on the
       Fisher-z scale, using only cleanly converged fits);
    3. refit every gene by generalised least squares (GLS) with that fixed
       block correlation, t-test on 13 - 3 = 10 residual df, BH.
  Why not just use the per-gene mixed model p-values: with 4 subjects
  contributing two samples, the subject variance of a single gene is
  barely estimable. In practice most per-gene fits fail to converge or sit
  at the boundary (variance = 0), and those fits give wildly
  anti-conservative p-values. Borrowing the correlation across genes is
  stable. The per-gene LMM results are kept in the table as diagnostics.
  The group coefficient is the case-control difference averaged over
  timepoints, adjusted for timepoint.

For comparison the script also fits:
  * naive OLS on all samples ignoring subject (the wrong model), to show
    what pseudoreplication does to the p-values;
  * OLS on the baseline samples only, on the same log scale.

Also writes a within-subject table of composition score changes for the
4 people with two samples (descriptive only).

Usage:
    python 07_extension_repeated_measures.py              # all filtered genes (~5-15 min)
    python 07_extension_repeated_measures.py --max-genes 2000   # quick run
"""

from __future__ import annotations

import argparse
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning

import config as C


def fit_gene(y: pd.Series, design: pd.DataFrame) -> dict:
    d = design.assign(y=y.values)
    out = {}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            m = smf.mixedlm("y ~ is_case + is_followup", d, groups=d["subject"]).fit(
                reml=True, method=["lbfgs", "powell"])
            sv = float(m.cov_re.iloc[0, 0])
            out.update(lmm_coef=m.params["is_case"], lmm_se=m.bse["is_case"],
                       lmm_p=m.pvalues["is_case"],
                       lmm_subject_var=sv, lmm_icc=sv / (sv + float(m.scale)),
                       lmm_converged=bool(m.converged)
                       and not any(issubclass(x.category, ConvergenceWarning) for x in w))
        except Exception:
            out.update(lmm_coef=np.nan, lmm_se=np.nan, lmm_p=np.nan,
                       lmm_subject_var=np.nan, lmm_icc=np.nan, lmm_converged=False)
    naive = smf.ols("y ~ is_case + is_followup", d).fit()
    out.update(naive_coef=naive.params["is_case"], naive_p=naive.pvalues["is_case"])
    base = d[d["is_followup"] == 0]
    b = smf.ols("y ~ is_case", base).fit()
    out.update(baseline_coef=b.params["is_case"], baseline_p=b.pvalues["is_case"])
    return out


def consensus_correlation(icc: pd.Series) -> float:
    """Trimmed mean (15% each end) of Fisher-z transformed ICCs, back-
    transformed; the approach of limma::duplicateCorrelation."""
    z = np.arctanh(icc.dropna().clip(0, 0.99))
    return float(np.tanh(stats.trim_mean(z, 0.15)))


def gls_fit(lognorm: pd.DataFrame, design: pd.DataFrame, rho: float) -> pd.DataFrame:
    X = sm.add_constant(design[["is_case", "is_followup"]].astype(float))
    subj = design["subject"].to_numpy(dtype=str)
    sigma = np.where(subj[:, None] == subj[None, :], rho, 0.0)
    np.fill_diagonal(sigma, 1.0)
    rows = {}
    for gene, y in lognorm.iterrows():
        f = sm.GLS(y.values, X.values, sigma=sigma).fit()
        rows[gene] = {"gls_coef": f.params[1], "gls_se": f.bse[1],
                      "gls_t": f.tvalues[1], "gls_p": f.pvalues[1]}
    return pd.DataFrame.from_dict(rows, orient="index")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-genes", type=int, default=None,
                    help="only fit the N most highly expressed genes (for a quick run)")
    args = ap.parse_args()

    C.ensure_dirs()
    samples = C.load_samples()          # all samples that passed QC
    counts, ann = C.load_counts(samples)
    filt = C.filter_genes(counts)
    lognorm = C.log_norm(filt)
    if args.max_genes:
        lognorm = lognorm.loc[filt.mean(axis=1).nlargest(args.max_genes).index]

    design = pd.DataFrame({
        "is_case": (samples["group"] == "case").astype(int),
        "is_followup": (samples["timepoint"] == "followup").astype(int),
        "subject": samples["subject"],
    }, index=samples.index)
    print(f"Design ({len(samples)} samples, {samples.subject.nunique()} subjects):")
    print(pd.crosstab(samples["group"], samples["timepoint"]).to_string())
    print(f"\nFitting {len(lognorm):,} genes ...")

    rows = {}
    for i, (gene, y) in enumerate(lognorm.iterrows(), 1):
        rows[gene] = fit_gene(y, design)
        if i % 1000 == 0:
            print(f"  {i:,} genes done")
    res = pd.DataFrame.from_dict(rows, orient="index")
    conv = res["lmm_converged"].astype(bool)
    rho = consensus_correlation(res.loc[conv, "lmm_icc"].astype(float))
    print(f"Consensus within-subject correlation: {rho:.3f} "
          f"(from {int(conv.sum()):,} cleanly converged per-gene fits)")
    res = gls_fit(lognorm, design, rho).join(res)
    res.insert(0, "gene_name", ann.loc[res.index, "gene_name"])
    for col in ("gls", "lmm", "naive", "baseline"):
        res[f"{col}_padj"] = C.bh(res[f"{col}_p"].astype(float))
    res = res.sort_values("gls_p")
    res.to_csv(C.TABLES / "07_repeated_measures_results.tsv", sep="\t")

    zero_var = (res["lmm_subject_var"].astype(float) < 1e-6).sum()
    summary = {
        "genes fitted": len(res),
        "consensus within-subject correlation (rho)": round(rho, 3),
        "padj<0.05, GLS with consensus rho (primary model)": int((res["gls_padj"] < 0.05).sum()),
        "padj<0.05, naive OLS all samples (pseudoreplication)": int((res["naive_padj"] < 0.05).sum()),
        "padj<0.05, OLS baseline only": int((res["baseline_padj"] < 0.05).sum()),
        "p<0.01, GLS": int((res["gls_p"] < 0.01).sum()),
        "p<0.01, naive": int((res["naive_p"] < 0.01).sum()),
        "p<0.01, baseline only": int((res["baseline_p"] < 0.01).sum()),
        "median p ratio naive / GLS": round(float((res["naive_p"] / res["gls_p"]).median()), 3),
        "correlation of group effect, GLS vs baseline-only": round(
            float(res["gls_coef"].corr(res["baseline_coef"].astype(float))), 3),
        "diagnostic: per-gene LMM converged cleanly": int(res["lmm_converged"].astype(bool).sum()),
        "diagnostic: per-gene LMM boundary fits (subject var ~0)": int(zero_var),
        "diagnostic: padj<0.05 from per-gene LMM (unreliable)": int((res["lmm_padj"] < 0.05).sum()),
    }
    pd.Series(summary).to_csv(C.TABLES / "07_summary.tsv", sep="\t", header=False)
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # ---- Within-subject composition change (descriptive) -------------------
    scores, _ = C.marker_scores(C.log_norm(filt), ann)
    paired = samples[samples["subject"].duplicated(keep=False)]
    ch = []
    for subj, g in paired.groupby("subject"):
        b = g[g["timepoint"] == "baseline"].index[0]
        f = g[g["timepoint"] == "followup"].index[0]
        row = {"subject": subj, "group": g["group"].iat[0],
               "age_baseline": samples.loc[b, "age"], "age_followup": samples.loc[f, "age"]}
        for ct in ["neutrophil", "T_cell", "B_cell", "NK", "monocyte", "NLR_proxy"]:
            if ct in scores:
                row[f"delta_{ct}"] = scores.loc[f, ct] - scores.loc[b, ct]
        ch.append(row)
    ch = pd.DataFrame(ch)
    ch.round(3).to_csv(C.TABLES / "07_within_subject_composition_change.tsv", sep="\t", index=False)
    print("\nWithin-subject change in composition scores (follow-up minus baseline):")
    print(ch.round(2).to_string(index=False))

    # ---- Figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    lp = -np.log10(res["gls_p"].astype(float))
    npv = -np.log10(res["naive_p"].astype(float))
    axes[0].scatter(lp, npv, s=3, alpha=0.4, c="grey")
    m = max(lp.max(), npv.max())
    axes[0].plot([0, m], [0, m], "k--", lw=0.8)
    axes[0].set_xlabel(f"-log10 p, GLS with within-subject correlation {rho:.2f}")
    axes[0].set_ylabel("-log10 p, naive OLS (samples treated as independent)")
    axes[0].set_title("Accounting for repeated measures vs ignoring them")
    axes[1].hist(res["gls_p"].astype(float).dropna(), bins=40, color="grey", edgecolor="white")
    axes[1].set_xlabel("p-value (GLS, group)")
    axes[1].set_title("p-value distribution, all-sample model")
    fig.tight_layout()
    fig.savefig(C.FIGS / "07_repeated_measures.png", dpi=200)
    print(f"\nFigure written to {C.FIGS}")


if __name__ == "__main__":
    main()
