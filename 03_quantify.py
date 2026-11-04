"""
03_quantify.py

Step 3: transcript quantification with Salmon and aggregation to gene level.

Choices (justify these in the report):
  * Salmon (selective alignment) rather than a genome aligner such as STAR:
    accurate gene-level estimates for DE, models multimapping reads
    probabilistically, corrects sequence and GC bias, and a human index plus
    12 to 16 samples at 2M pairs runs on a laptop. STAR needs about 32 GB RAM.
  * GENCODE v44 comprehensive transcript set: includes all annotated
    transcripts (incl. non-coding), consistent with Ensembl 110 gene IDs.
  * Transcriptome-only index by default. A decoy-aware index (genome as
    decoy) reduces false mapping of intronic/intergenic reads but needs
    roughly 18 GB RAM to build; use --decoys if your machine allows.
  * Gene counts = sum of transcript NumReads per gene (equivalent to
    tximport countsFromAbundance = "no"), rounded for DESeq2.

Outputs:
  results/salmon/<run>/quant.sf
  results/counts/gene_counts.tsv       genes x samples (estimated counts)
  results/counts/gene_tpm.tsv          genes x samples (TPM)
  results/counts/gene_annotation.tsv   gene_id, gene_name, gene_type
  results/qc/salmon_metrics.tsv        mapping rate, library type per sample

Usage:
    python 03_quantify.py
    python 03_quantify.py --decoys      # build decoy-aware index (more RAM)
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil

import pandas as pd
import requests

import config as C

TX_FA = C.REF / f"gencode.v{C.GENCODE_RELEASE}.transcripts.fa.gz"
GENOME_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/"
    f"release_{C.GENCODE_RELEASE}/GRCh38.primary_assembly.genome.fa.gz"
)
GENOME_FA = C.REF / "GRCh38.primary_assembly.genome.fa.gz"


def fetch(url, dest):
    if dest.exists():
        return
    print(f"  downloading {url}")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            shutil.copyfileobj(r.raw, fh, length=8 * 1024 * 1024)


def tx2gene_from_fasta(fa) -> pd.DataFrame:
    """GENCODE headers look like:
    >ENST..|ENSG..|OTTHUMG..|OTTHUMT..|TX-NAME|GENE_NAME|length|biotype|"""
    rows = []
    with gzip.open(fa, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                f = line[1:].strip().split("|")
                rows.append((f[0], f[1], f[5], f[7]))
    return pd.DataFrame(rows, columns=["transcript_id", "gene_id", "gene_name", "transcript_type"])


def build_index(decoys: bool) -> "C.Path":
    idx = C.REF / ("salmon_index_decoy" if decoys else "salmon_index")
    if (idx / "versionInfo.json").exists():
        return idx
    fetch(C.GENCODE_TX_URL, TX_FA)
    cmd = ["salmon", "index", "-t", TX_FA, "-i", idx, "-k", "31", "--gencode", "-p", str(C.THREADS)]
    if decoys:
        fetch(GENOME_URL, GENOME_FA)
        decoy_txt = C.REF / "decoys.txt"
        with gzip.open(GENOME_FA, "rt") as fh, open(decoy_txt, "w") as out:
            for line in fh:
                if line.startswith(">"):
                    out.write(line[1:].split()[0] + "\n")
        gentrome = C.REF / "gentrome.fa.gz"
        with open(gentrome, "wb") as out:
            for f in (TX_FA, GENOME_FA):
                with open(f, "rb") as src:
                    shutil.copyfileobj(src, out)
        cmd = ["salmon", "index", "-t", gentrome, "-d", decoy_txt, "-i", idx,
               "-k", "31", "--gencode", "-p", str(C.THREADS)]
    C.run(cmd, "03_salmon_index")
    return idx


def quantify(idx, run: str) -> None:
    out = C.QUANT / run
    if (out / "quant.sf").exists():
        return
    t1, t2 = C.TRIM / f"{run}_trim_1.fastq.gz", C.TRIM / f"{run}_trim_2.fastq.gz"
    C.run(["salmon", "quant", "-i", idx, "-l", "A", "-1", t1, "-2", t2,
           "-p", str(C.THREADS), "--validateMappings", "--seqBias", "--gcBias",
           "-o", out], "03_salmon_quant")


def aggregate(runs: list[str], tx2gene: pd.DataFrame) -> None:
    t2g = tx2gene.set_index("transcript_id")["gene_id"]
    counts, tpm, metrics = {}, {}, []
    for run in runs:
        q = pd.read_csv(C.QUANT / run / "quant.sf", sep="\t", index_col=0)
        genes = t2g.reindex(q.index)
        if genes.isna().any():
            print(f"  WARNING {run}: {genes.isna().sum()} transcripts not in tx2gene")
        counts[run] = q["NumReads"].groupby(genes).sum()
        tpm[run] = q["TPM"].groupby(genes).sum()
        meta = json.loads((C.QUANT / run / "aux_info" / "meta_info.json").read_text())
        lib = json.loads((C.QUANT / run / "lib_format_counts.json").read_text())
        metrics.append({"run": run,
                        "salmon_processed": meta["num_processed"],
                        "salmon_mapped": meta["num_mapped"],
                        "salmon_pct_mapped": meta["percent_mapped"],
                        "library_type": lib.get("expected_format", "NA")})

    counts = pd.DataFrame(counts).fillna(0)
    tpm = pd.DataFrame(tpm).fillna(0)
    counts.index.name = tpm.index.name = "gene_id"
    counts.round(3).to_csv(C.COUNTS / "gene_counts.tsv", sep="\t")
    tpm.round(4).to_csv(C.COUNTS / "gene_tpm.tsv", sep="\t")

    # Gene annotation. The transcript FASTA has no gene biotype, so gene_type
    # is the most common transcript biotype of that gene (used only to flag
    # rRNA and mitochondrial genes in QC).
    ann = (tx2gene.groupby("gene_id")
           .agg(gene_name=("gene_name", "first"),
                gene_type=("transcript_type", lambda s: s.mode().iat[0]))
           .reindex(counts.index))
    ann.to_csv(C.COUNTS / "gene_annotation.tsv", sep="\t")

    m = pd.DataFrame(metrics).set_index("run")
    m.to_csv(C.QC / "salmon_metrics.tsv", sep="\t")
    print(m.to_string())
    print(f"\nGene matrix: {counts.shape[0]:,} genes x {counts.shape[1]} samples")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoys", action="store_true")
    args = ap.parse_args()

    C.ensure_dirs()
    samples = C.load_samples(include_excluded=True)
    with open(C.LOGS / "03_versions.txt", "w") as fh:
        fh.write(f"salmon: {C.tool_version(['salmon', '--version'])}\n")
        fh.write(f"reference: {C.GENCODE_TX_URL}\n")
        fh.write(f"decoy-aware index: {args.decoys}\n")

    idx = build_index(args.decoys)
    for run in samples.index:
        print(f"=== {run} ===")
        quantify(idx, run)

    aggregate(samples.index.tolist(), tx2gene_from_fasta(TX_FA))

    # MultiQC report across FastQC, fastp and Salmon
    C.run(["multiqc", "-f", "-o", C.QC / "multiqc", C.QC, C.QUANT], "03_multiqc")


if __name__ == "__main__":
    main()
