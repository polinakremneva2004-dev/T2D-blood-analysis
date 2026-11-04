"""
config.py

Shared settings, paths and helper functions for every step of the
GSE184050 / PRJNA762935 whole-blood RNA-seq analysis.

Every numbered script imports from here, so parameters are set in one place
and can be quoted directly in the report.
"""

from __future__ import annotations

import datetime as dt
import shlex
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Parameters (quote these in the methods section)
# -----------------------------------------------------------------------------
PROJECT = "PRJNA762935"
NREADS = 2_000_000          # read pairs kept per sample after subsampling
SEED = 100                  # seqtk random seed
THREADS = 4                 # adjust to your laptop

GENCODE_RELEASE = 44
GENCODE_TX_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/"
    f"release_{GENCODE_RELEASE}/gencode.v{GENCODE_RELEASE}.transcripts.fa.gz"
)

# Samples excluded from all expression analyses after QC (decided from the
# step 2 to 4 QC output of the first full run; see 04_sample_qc.tsv and
# 04_library_qc.png). All three show the same failure pattern, different from
# the other 13 libraries:
#   * Salmon infers an UNSTRANDED library (IU) where all others are stranded
#     (ISR), i.e. a different or failed library prep
#   * no ribosomal or globin depletion: ~96% of reads fall in ~12 genes
#     (mitochondrial and nuclear rRNA, an rRNA-like lncRNA, HBB, HBA2)
#   * GC 56% vs 43-50%, duplication 42-53% vs 1-17%
#   * only 430 to 2,720 genes with >= 10 reads, vs 8,100 to 9,800
#   * sample-sample correlation outliers (robust z -4.8 to -12)
# They still appear in the QC tables and figures, flagged.
EXCLUDE = {
    "SRR15881883": "case_13: failed/unstranded library, rRNA+globin dominated",
    "SRR15881953": "control_33: failed/unstranded library, rRNA+globin dominated",
    "SRR15881878": "case_8: failed/unstranded library, rRNA+globin dominated",
}

# Gene filter used before normalisation, PCA and DE:
# keep genes with at least MIN_COUNT reads in at least MIN_SAMPLES samples.
# MIN_SAMPLES = size of the smallest group after exclusions (4 cases), so a
# gene expressed in only one group can still pass.
MIN_COUNT = 10
MIN_SAMPLES = 4

N_TOP_VAR_GENES = 500       # genes used for PCA (as in DESeq2 plotPCA)
FDR = 0.05

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
META = ROOT / "metadata"
DATA = ROOT / "data"
RAW = DATA / "raw"
SUB = DATA / "subsampled"
TRIM = DATA / "trimmed"
REF = DATA / "reference"
RESULTS = ROOT / "results"
QC = RESULTS / "qc"
QUANT = RESULTS / "salmon"
COUNTS = RESULTS / "counts"
TABLES = RESULTS / "tables"
FIGS = RESULTS / "figures"
LOGS = ROOT / "logs"

SAMPLES_TSV = META / "samples.tsv"


def ensure_dirs() -> None:
    for d in (META, RAW, SUB, TRIM, REF, QC, QUANT, COUNTS, TABLES, FIGS, LOGS):
        d.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Sample sheet
# -----------------------------------------------------------------------------
def load_samples(task: str | None = None, include_excluded: bool = False) -> pd.DataFrame:
    """Load metadata/samples.tsv. task='primary' gives the baseline samples,
    task=None gives all. Samples in EXCLUDE are dropped unless
    include_excluded=True (used for QC tables and for steps 1 to 3)."""
    s = pd.read_csv(SAMPLES_TSV, sep="\t")
    if task is not None:
        s = s[s["task"] == task]
    s = s.set_index("run")
    s["excluded"] = s.index.isin(list(EXCLUDE))
    if not include_excluded:
        s = s[~s["excluded"]]
    return s


# -----------------------------------------------------------------------------
# Running external tools
# -----------------------------------------------------------------------------
def run(cmd: list[str], log_name: str | None = None) -> None:
    """Run a command, stop on failure, and append the command plus its output
    to logs/<log_name>.log so every parameter used is on record."""
    print("  $ " + " ".join(shlex.quote(str(c)) for c in cmd))
    res = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if log_name:
        with open(LOGS / f"{log_name}.log", "a") as fh:
            fh.write(f"\n### {dt.datetime.now().isoformat()}\n$ {' '.join(map(str, cmd))}\n")
            fh.write(res.stdout)
            fh.write(res.stderr)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {cmd}\n{res.stderr[-2000:]}")


def tool_version(cmd: list[str]) -> str:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        out = (res.stdout + res.stderr).strip().splitlines()
        return out[0] if out else "unknown"
    except FileNotFoundError:
        return "NOT INSTALLED"


# -----------------------------------------------------------------------------
# Expression helpers shared by steps 4 to 7
# -----------------------------------------------------------------------------
def load_counts(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (counts genes x samples, gene annotation) restricted to the
    given samples, in sample-sheet order."""
    counts = pd.read_csv(COUNTS / "gene_counts.tsv", sep="\t", index_col=0)
    ann = pd.read_csv(COUNTS / "gene_annotation.tsv", sep="\t", index_col=0)
    counts = counts[samples.index.tolist()].round().astype(int)
    return counts, ann.loc[counts.index]


def filter_genes(counts: pd.DataFrame, min_count=MIN_COUNT, min_samples=MIN_SAMPLES) -> pd.DataFrame:
    keep = (counts >= min_count).sum(axis=1) >= min_samples
    return counts.loc[keep]


def size_factors(counts: pd.DataFrame) -> pd.Series:
    """DESeq2 median-of-ratios size factors (Anders and Huber 2010)."""
    log_c = np.log(counts.replace(0, np.nan))
    log_geo = log_c.mean(axis=1)
    usable = log_c.notna().all(axis=1)
    ratios = log_c.loc[usable].sub(log_geo[usable], axis=0)
    return np.exp(ratios.median(axis=0))


def log_norm(counts: pd.DataFrame) -> pd.DataFrame:
    """log2(normalised count + 1) using median-of-ratios size factors."""
    sf = size_factors(counts)
    return np.log2(counts.div(sf, axis=1) + 1)


def bh(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values, NaN-safe."""
    from statsmodels.stats.multitest import multipletests
    out = pd.Series(np.nan, index=p.index)
    ok = p.notna()
    if ok.any():
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out


# -----------------------------------------------------------------------------
# Blood cell-type marker genes for composition scores (step 6, also used in
# step 4 to annotate principal components).
#
# Chosen as genes with high, specific expression in one circulating cell type
# in sorted-cell reference data (e.g. the Human Protein Atlas blood atlas,
# Monaco et al. 2019 Cell Reports). Assumption: their expression per cell is
# roughly constant, so their bulk level tracks the abundance of that cell type.
# Step 6 checks this by testing whether markers within a set actually
# correlate with each other.
# -----------------------------------------------------------------------------
CELL_MARKERS: dict[str, list[str]] = {
    "neutrophil": ["FCGR3B", "CXCR2", "CSF3R", "FPR1", "CXCR1", "MME", "CMTM2"],
    "monocyte":   ["CD14", "CSF1R", "FCN1", "VCAN", "LYZ", "S100A12"],
    "T_cell":     ["CD3E", "CD3D", "CD3G", "CD2", "CD5", "CD6", "LCK"],
    "CD8_T":      ["CD8A", "CD8B"],
    "B_cell":     ["MS4A1", "CD19", "CD79A", "CD79B", "BANK1", "FCRLA"],
    "NK":         ["NCAM1", "KLRF1", "KLRD1", "GNLY", "NKG7", "SH2D1B"],
    "eosinophil": ["CLC", "IL5RA", "CCR3", "SIGLEC8", "PRSS33"],
    "platelet":   ["PPBP", "PF4", "ITGA2B", "GP9", "GP1BA", "TUBB1"],
    "erythroid":  ["ALAS2", "CA1", "SLC4A1", "AHSP", "HBD"],
}

GLOBIN_GENES = ["HBA1", "HBA2", "HBB", "HBD", "HBG1", "HBG2", "HBM", "HBQ1", "HBZ", "HBE1"]
SEX_GENES_Y = ["RPS4Y1", "DDX3Y", "UTY", "KDM5D", "EIF1AY", "ZFY"]


def marker_scores(lognorm: pd.DataFrame, ann: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-sample score for each cell type: mean z-score (across samples) of
    that type's marker genes on the log scale. Scores are relative between
    samples, not absolute proportions.

    Returns (scores samples x cell types, coherence table)."""
    name_to_id = ann.reset_index().drop_duplicates("gene_name").set_index("gene_name")["gene_id"]
    z = lognorm.sub(lognorm.mean(axis=1), axis=0).div(lognorm.std(axis=1).replace(0, np.nan), axis=0)
    scores, rows = {}, []
    for ct, genes in CELL_MARKERS.items():
        ids = [name_to_id[g] for g in genes if g in name_to_id.index and name_to_id[g] in z.index]
        found = [g for g in genes if g in name_to_id.index and name_to_id[g] in z.index]
        if not ids:
            rows.append({"cell_type": ct, "n_markers_found": 0, "markers": "", "mean_pairwise_r": np.nan})
            continue
        sub = z.loc[ids].dropna()
        scores[ct] = sub.mean(axis=0)
        if len(sub) > 1:
            r = np.corrcoef(lognorm.loc[sub.index].values)
            mean_r = r[np.triu_indices_from(r, k=1)].mean()
        else:
            mean_r = np.nan
        rows.append({"cell_type": ct, "n_markers_found": len(ids),
                     "markers": ",".join(found), "mean_pairwise_r": mean_r})
    scores = pd.DataFrame(scores)
    # Single summary axis analogous to a neutrophil-to-lymphocyte ratio
    lymph = [c for c in ("T_cell", "B_cell", "NK") if c in scores]
    if "neutrophil" in scores and lymph:
        scores["NLR_proxy"] = scores["neutrophil"] - scores[lymph].mean(axis=1)
    return scores, pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# DESeq2 (PyDESeq2) wrapper used by steps 5 and 6
# -----------------------------------------------------------------------------
def run_deseq(counts: pd.DataFrame, meta: pd.DataFrame, factors: list[str],
              continuous: list[str] | None = None) -> pd.DataFrame:
    """Negative binomial GLM (PyDESeq2), Wald test for case vs control.

    counts: genes x samples (filtered). meta: samples x covariates with a
    'group' column holding 'case'/'control'. Continuous covariates are
    standardised (mean 0, SD 1) so their coefficients are comparable.
    Design formula, e.g. "~ group" or "~ group + nlr" (PyDESeq2 >= 0.5).
    """
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    md = meta[factors].copy()
    for c in continuous or []:
        md[c] = (md[c] - md[c].mean()) / md[c].std()
    inference = DefaultInference(n_cpus=THREADS)
    dds = DeseqDataSet(counts=counts.T, metadata=md, design="~ " + " + ".join(factors),
                       refit_cooks=True, inference=inference, quiet=True)
    dds.deseq2()
    ds = DeseqStats(dds, contrast=["group", "case", "control"], alpha=FDR,
                    cooks_filter=True, independent_filter=True,
                    inference=inference, quiet=True)
    ds.summary()
    return ds.results_df
