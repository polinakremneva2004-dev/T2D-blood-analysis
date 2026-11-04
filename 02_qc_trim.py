"""
02_qc_trim.py

Step 2: read-level quality control and trimming of the subsampled FASTQs.

  1. FastQC on the subsampled reads (before trimming)
  2. fastp: adapter and quality trimming, plus per-sample QC metrics
  3. Parse fastp JSON into results/qc/read_qc_metrics.tsv

Notes for the report:
  * Duplication in whole-blood RNA-seq is expected to be high, because a few
    genes (globins, some neutrophil transcripts) take up a large share of
    reads. High duplication here is mostly biological, not PCR artefact, so
    reads are NOT deduplicated.
  * HiSeq 2000 uses four-colour chemistry, so poly-G trimming (needed for
    NovaSeq/NextSeq) is switched off explicitly.

Usage:
    python 02_qc_trim.py
"""

from __future__ import annotations

import json

import pandas as pd

import config as C

FASTP_ARGS = [
    "--detect_adapter_for_pe",      # find adapters by overlap analysis
    "--qualified_quality_phred", "20",
    "--unqualified_percent_limit", "40",
    "--cut_tail", "--cut_tail_mean_quality", "20",
    "--length_required", "36",
    "--disable_trim_poly_g",
]


def parse_fastp(js: dict, run: str) -> dict:
    s = js["summary"]
    before, after = s["before_filtering"], s["after_filtering"]
    filt = js["filtering_result"]
    return {
        "run": run,
        "pairs_in": before["total_reads"] // 2,
        "pairs_out": after["total_reads"] // 2,
        "pct_pairs_kept": 100 * after["total_reads"] / before["total_reads"],
        "q30_before": 100 * before["q30_rate"],
        "q30_after": 100 * after["q30_rate"],
        "gc_pct": 100 * before["gc_content"],
        "mean_len_r1_before": before["read1_mean_length"],
        "mean_len_r1_after": after["read1_mean_length"],
        "pct_reads_adapter_trimmed": 100 * js.get("adapter_cutting", {}).get("adapter_trimmed_reads", 0)
                                      / before["total_reads"],
        "pct_low_quality": 100 * filt["low_quality_reads"] / before["total_reads"],
        "pct_too_short": 100 * filt["too_short_reads"] / before["total_reads"],
        "duplication_pct": 100 * js.get("duplication", {}).get("rate", float("nan")),
        "insert_size_peak": js.get("insert_size", {}).get("peak", float("nan")),
    }


def main() -> None:
    C.ensure_dirs()
    samples = C.load_samples(include_excluded=True)
    fastqc_dir = C.QC / "fastqc_raw"
    fastp_dir = C.QC / "fastp"
    fastqc_dir.mkdir(exist_ok=True)
    fastp_dir.mkdir(exist_ok=True)

    with open(C.LOGS / "02_versions.txt", "w") as fh:
        fh.write(f"fastqc: {C.tool_version(['fastqc', '--version'])}\n")
        fh.write(f"fastp: {C.tool_version(['fastp', '--version'])}\n")
        fh.write("fastp args: " + " ".join(FASTP_ARGS) + "\n")

    rows = []
    for run in samples.index:
        print(f"=== {run} ===")
        r1, r2 = C.SUB / f"{run}_sub_1.fastq.gz", C.SUB / f"{run}_sub_2.fastq.gz"
        t1, t2 = C.TRIM / f"{run}_trim_1.fastq.gz", C.TRIM / f"{run}_trim_2.fastq.gz"
        js_path = fastp_dir / f"{run}.fastp.json"

        if not (fastqc_dir / f"{run}_sub_1_fastqc.zip").exists():
            C.run(["fastqc", "-q", "-t", str(C.THREADS), "-o", fastqc_dir, r1, r2], "02_fastqc")

        if not js_path.exists():
            C.run(["fastp", "-i", r1, "-I", r2, "-o", t1, "-O", t2,
                   "--thread", str(C.THREADS), *FASTP_ARGS,
                   "--json", js_path, "--html", fastp_dir / f"{run}.fastp.html",
                   "--report_title", run], "02_fastp")

        rows.append(parse_fastp(json.loads(js_path.read_text()), run))

    qc = pd.DataFrame(rows).set_index("run")
    qc = samples[["group", "subject", "timepoint", "reads_brief"]].join(qc)
    qc.round(2).to_csv(C.QC / "read_qc_metrics.tsv", sep="\t")
    print(qc.round(1).to_string())

    # Simple flags to look at, not automatic exclusions
    flags = []
    for run, r in qc.iterrows():
        if r["q30_before"] < 80:
            flags.append(f"{run}: Q30 {r['q30_before']:.1f}% (<80%)")
        if r["pct_pairs_kept"] < 90:
            flags.append(f"{run}: only {r['pct_pairs_kept']:.1f}% of pairs pass filtering")
        if r["pct_reads_adapter_trimmed"] > 20:
            flags.append(f"{run}: {r['pct_reads_adapter_trimmed']:.1f}% reads adapter-trimmed (short inserts?)")
    print("\nQC flags:\n  " + ("\n  ".join(flags) if flags else "none"))


if __name__ == "__main__":
    main()
