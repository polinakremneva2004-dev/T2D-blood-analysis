"""
01_download_subsample.py

Step 1: obtain raw FASTQ files from ENA and subsample them.

  1. Query the ENA portal API for FASTQ locations, read counts and MD5s
  2. Download the paired FASTQs for each of the 16 runs
  3. Verify MD5 checksums (computed while downloading)
  4. Subsample to NREADS read pairs with seqtk (reservoir sampling, fixed seed)
  5. Check that R1 and R2 read IDs still match line for line
  6. Write results/qc/subsample_summary.tsv

Usage:
    python 01_download_subsample.py            # deletes full-depth files after subsampling
    python 01_download_subsample.py --keep-full

Re-running skips samples that are already subsampled.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import subprocess
import time
from pathlib import Path

import pandas as pd
import requests  # ENA API query; FASTQ downloads use wget (resumable)

import config as C

ENA_API = "https://www.ebi.ac.uk/ena/portal/api/filereport"


def ena_report(runs: list[str]) -> pd.DataFrame:
    params = {"accession": C.PROJECT, "result": "read_run",
              "fields": "run_accession,read_count,fastq_ftp,fastq_md5", "format": "tsv"}
    r = requests.get(ENA_API, params=params, timeout=120)
    r.raise_for_status()
    rep = pd.read_csv(io.StringIO(r.text), sep="\t")
    rep.to_csv(C.META / "ena_filereport_full.tsv", sep="\t", index=False)
    rep = rep[rep["run_accession"].isin(runs)].set_index("run_accession")
    missing = set(runs) - set(rep.index)
    if missing:
        raise SystemExit(f"Runs missing from ENA report: {sorted(missing)}")
    rep.to_csv(C.META / "ena_filereport_selected.tsv", sep="\t")
    return rep


def paired_files(ftp: str, md5: str) -> list[tuple[str, str]]:
    """Return [(url_R1, md5_R1), (url_R2, md5_R2)], ignoring any unpaired file."""
    pairs = dict(zip(ftp.split(";"), md5.split(";")))
    r1 = [(u, m) for u, m in pairs.items() if u.endswith("_1.fastq.gz")]
    r2 = [(u, m) for u, m in pairs.items() if u.endswith("_2.fastq.gz")]
    if len(r1) != 1 or len(r2) != 1:
        raise SystemExit(f"Could not identify paired files in: {ftp}")
    return [r1[0], r2[0]]


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, expected_md5: str, retries: int = 20) -> None:
    """Download with wget in resume mode (-c): if the connection drops, the
    next attempt continues from the bytes already on disk instead of starting
    again. The finished file is then checked against ENA's MD5; a corrupt
    file is deleted and downloaded afresh."""
    if not url.startswith("http"):
        url = "https://" + url
    for attempt in range(1, retries + 1):
        rc = subprocess.call(["wget", "-c", "-q", "--show-progress",
                              "--tries=10", "--waitretry=30", "--read-timeout=120",
                              "--retry-connrefused", "-P", str(dest.parent), url])
        if rc == 0 and dest.exists():
            if file_md5(dest) == expected_md5:
                with open(C.LOGS / "01_md5.log", "a") as log:
                    log.write(f"{dest.name}\t{expected_md5}\tOK\n")
                return
            print(f"    MD5 mismatch for {dest.name}, deleting and downloading again")
            dest.unlink()
        else:
            print(f"    attempt {attempt} interrupted (wget exit code {rc}), "
                  f"resuming in {min(attempt, 5)} min")
            time.sleep(60 * min(attempt, 5))
    raise SystemExit(f"Giving up on {url}")


def seqtk_sample(src: Path, dest: Path) -> None:
    """seqtk with an integer N uses reservoir sampling, so reads are drawn
    uniformly from the whole file rather than taking the first N reads (which
    come from particular flowcell tiles). The same seed on R1 and R2 picks the
    same record positions, so mates stay paired."""
    with open(dest, "wb") as out:
        p1 = subprocess.Popen(["seqtk", "sample", f"-s{C.SEED}", str(src), str(C.NREADS)],
                              stdout=subprocess.PIPE)
        p2 = subprocess.Popen(["gzip", "-c"], stdin=p1.stdout, stdout=out)
        p1.stdout.close()
        p2.communicate()
        if p1.wait() != 0 or p2.returncode != 0:
            raise SystemExit(f"seqtk failed on {src}")


def read_id(header: str) -> str:
    rid = header.split()[0]
    return rid[:-2] if rid.endswith(("/1", "/2")) else rid


def check_pairs(f1: Path, f2: Path) -> tuple[int, int, bool]:
    """Count reads in each file and confirm IDs match record by record."""
    from itertools import zip_longest
    n1 = n2 = 0
    match = True
    with gzip.open(f1, "rt") as a, gzip.open(f2, "rt") as b:
        for i, (la, lb) in enumerate(zip_longest(a, b)):
            if i % 4 == 0:
                n1 += la is not None
                n2 += lb is not None
                if match and (la is None or lb is None or read_id(la) != read_id(lb)):
                    match = False
    return n1, n2, match


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-full", action="store_true", help="keep full-depth FASTQs")
    args = ap.parse_args()

    C.ensure_dirs()
    samples = C.load_samples(include_excluded=True)
    with open(C.LOGS / "01_versions.txt", "w") as fh:
        fh.write(f"seqtk: {C.tool_version(['seqtk'])}\n")
        fh.write(f"NREADS={C.NREADS} SEED={C.SEED}\n")

    rep = ena_report(samples.index.tolist())
    rows = []
    for run in samples.index:
        info = rep.loc[run]
        print(f"=== {run} (ENA read pairs: {info['read_count']:,}) ===")
        out1, out2 = C.SUB / f"{run}_sub_1.fastq.gz", C.SUB / f"{run}_sub_2.fastq.gz"

        if out1.exists() and out2.exists():
            print("  already subsampled, skipping download")
        else:
            (u1, m1), (u2, m2) = paired_files(info["fastq_ftp"], info["fastq_md5"])
            raw1, raw2 = C.RAW / f"{run}_1.fastq.gz", C.RAW / f"{run}_2.fastq.gz"
            for url, dest, md5 in ((u1, raw1, m1), (u2, raw2, m2)):
                print(f"  downloading {dest.name}")
                download(url, dest, md5)
            seqtk_sample(raw1, out1)
            seqtk_sample(raw2, out2)
            if not args.keep_full:
                raw1.unlink()
                raw2.unlink()

        n1, n2, ok = check_pairs(out1, out2)
        print(f"  R1={n1:,}  R2={n2:,}  paired={'yes' if ok else 'NO'}")
        if not ok:
            print("  WARNING: R1/R2 mismatch, investigate before continuing")
        rows.append({"run": run, "ena_read_pairs": info["read_count"],
                     "sub_r1": n1, "sub_r2": n2, "pairs_match": ok, "seed": C.SEED})

    summary = pd.DataFrame(rows)
    summary.to_csv(C.QC / "subsample_summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
