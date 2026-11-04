# Whole-blood RNA-seq, T2D converters vs non-converters (GSE184050)

Python pipeline from raw FASTQ to differential expression, cell-composition
analysis and a repeated-measures extension. Python orchestrates the standard
command-line tools (seqtk, FastQC, fastp, Salmon); all analysis is in Python.

## Setup

```bash
conda env create -f environment.yml      # Apple Silicon: prefix with CONDA\_SUBDIR=osx-64
conda activate t2d-blood
conda env export --no-builds > environment.lock.yml   # exact versions for the report
```

Edit `THREADS` in `config.py` to match your machine. All parameters live in `config.py`.

## Run order

|Step|Script|What it does|Main outputs|
|-|-|-|-|
|1|`01\_download\_subsample.py`|ENA download, MD5 check, seqtk subsample to 2M pairs (seed 100), pairing check|`data/subsampled/`, `results/qc/subsample\_summary.tsv`|
|2|`02\_qc\_trim.py`|FastQC, fastp trimming, read-level QC table|`results/qc/read\_qc\_metrics.tsv`|
|3|`03\_quantify.py`|Salmon index (GENCODE v44) and quant, gene-level aggregation, MultiQC|`results/counts/gene\_counts.tsv`, `results/qc/multiqc/`|
|4|`04\_explore.py`|Globin/mito/sex QC, normalisation, outliers, PCA, PC vs covariate correlations|`results/tables/04\_\*`, `results/figures/04\_\*`|
|5|`05\_differential\_expression.py`|PyDESeq2 `\~ group`, BH FDR, p-value histogram, power estimate|`results/tables/05\_\*`, `results/figures/05\_\*`|
|6|`06\_cell\_composition.py`|Marker-gene cell-type scores, group tests, DE adjusted for NLR proxy|`results/tables/06\_\*`, `results/figures/06\_\*`|
|7|`07\_extension\_repeated\_measures.py`|All QC-passing samples, GLS with consensus within-subject correlation|`results/tables/07\_\*`, `results/figures/07\_\*`|
|8|`08\_compare\_geo\_matrix.py`|Optional: agreement with the authors' GEO matrix|`results/tables/08\_\*`|

`python run\_all.py` runs steps 1 to 7 in order (`--from 4` restarts at step 4).
Step 6 also accepts `--signature LM22.txt` for NNLS deconvolution against a
published signature matrix.

Expected run time on a laptop: downloads take hours (about 40 to 60 GB in
total, deleted after subsampling unless `--keep-full`); the Salmon index
takes 20 to 40 minutes; everything after that takes minutes.

## Key design points (for the report)

* **What the comparison contrasts.** All 12 primary samples are baseline
draws. "Case" means the person later transitioned to T2D, so the primary
test is future converters vs non-converters at a pre-diagnostic timepoint.
* **Subsampling.** 2,000,000 read pairs per sample, seqtk reservoir sampling
with the same seed on R1 and R2. Equalises depth (originally 9.3M to 38.3M
pairs) but costs sensitivity for low-expressed genes; step 5 reports the
resulting detectable effect sizes.
* **QC exclusions.** Three baseline libraries (case\_13, case\_8, control\_33)
are excluded from all expression analyses: Salmon infers them as
unstranded where all others are stranded, \~96% of their reads fall in
\~12 rRNA/globin genes, and they detect only 430 to 2,720 genes. They
stay in the QC tables and `04\_library\_qc.png`. The primary comparison
is therefore 4 cases vs 5 controls. See `EXCLUDE` in `config.py`.
* **Composition.** Marker scores are relative, assume constant per-cell
marker expression, and are checked for internal coherence. Only one
composition covariate (NLR proxy) is added, because 9 samples leave few
residual degrees of freedom.
* **Repeated measures.** Four subjects contribute two samples in the
extension. A consensus within-subject correlation is estimated across
genes (as in limma's duplicateCorrelation) and each gene is fitted by GLS
with that correlation; per-gene mixed models are too unstable with this
few repeated subjects. The naive model is fitted alongside to show the
difference.

## Use of AI tools

This code was drafted with the help of Claude (Anthropic) and reviewed,
run and checked by me.

