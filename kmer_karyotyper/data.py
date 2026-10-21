"""
Data loading and sequence extraction module for FASTQ, CRAM, BAM, and SAM files,
as well as fetching 1000 Genomes data or remote URLs.
"""

import gzip
import os
import urllib.request
from typing import Generator, Iterable, List, Optional, Union, Tuple
import pysam


def is_url(path_or_url: str) -> bool:
    """Check if a string is an HTTP, HTTPS, or FTP URL."""
    return (
        path_or_url.startswith("http://")
        or path_or_url.startswith("https://")
        or path_or_url.startswith("ftp://")
    )


def download_file(url: str, output_path: str) -> str:
    """
    Download a file from a URL to a local destination.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(output_path, "wb") as out_file:
        out_file.write(response.read())
    return output_path


def get_1000_genomes_url(sample_id: str, file_type: str = "fastq") -> str:
    """
    Resolve or construct 1000 Genomes data URL for a given sample ID.
    Supports known standard test samples or general ENA/1000G path construction.
    """
    known_urls = {
        "HG00096": {
            "fastq": "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR062/SRR062634/SRR062634_1.fastq.gz",
            "cram": "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/data/ERR3239276/HG00096.final.cram"
        },
        "HG00097": {
            "fastq": "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR062/SRR062635/SRR062635_1.fastq.gz",
            "cram": "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/data/ERR3239277/HG00097.final.cram"
        },
        "HG00099": {
            "fastq": "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR062/SRR062637/SRR062637_1.fastq.gz",
        },
        "HG00100": {
            "fastq": "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR062/SRR062641/SRR062641_1.fastq.gz",
        }
    }

    if sample_id in known_urls and file_type in known_urls[sample_id]:
        return known_urls[sample_id][file_type]

    if is_url(sample_id):
        return sample_id

    raise ValueError(f"1000 Genomes URL for sample {sample_id} ({file_type}) not found in registry. Please specify a direct URL or path.")


def read_fastq_sequences(
    filepath_or_url: str,
    max_reads: Optional[int] = None,
    cache_dir: Optional[str] = None
) -> Generator[str, None, None]:
    """
    Yield nucleotide sequences from a FASTQ or FASTQ.GZ file or URL.
    """
    local_path = filepath_or_url

    if is_url(filepath_or_url):
        filename = os.path.basename(filepath_or_url.split("?")[0])
        cache_dir = cache_dir or "/tmp/kmer_karyotyper_cache"
        local_path = os.path.join(cache_dir, filename)
        if not os.path.exists(local_path):
            download_file(filepath_or_url, local_path)

    if local_path.endswith(".gz"):
        handle = gzip.open(local_path, "rt")
    else:
        handle = open(local_path, "rt")

    count = 0
    with handle as f:
        while True:
            header = f.readline()
            if not header:
                break
            seq = f.readline().strip()
            f.readline()  # separator '+'
            f.readline()  # quality string
            if seq:
                yield seq
                count += 1
                if max_reads is not None and count >= max_reads:
                    break


def read_cram_sequences(
    filepath_or_url: str,
    max_reads: Optional[int] = None,
    fasta_reference: Optional[str] = None
) -> Generator[str, None, None]:
    """
    Yield read sequences from a CRAM, BAM, or SAM file or URL using pysam.
    """
    mode = "rc" if filepath_or_url.endswith(".cram") else "rb" if filepath_or_url.endswith(".bam") else "r"
    save_args = {}
    if fasta_reference:
        save_args["reference_filename"] = fasta_reference

    alignment_file = pysam.AlignmentFile(filepath_or_url, mode, check_sq=False, **save_args)
    count = 0
    try:
        for read in alignment_file.fetch(until_eof=True):
            seq = read.query_sequence
            if seq:
                yield seq
                count += 1
                if max_reads is not None and count >= max_reads:
                    break
    finally:
        alignment_file.close()


def read_sequences(
    source: str,
    max_reads: Optional[int] = None,
    fasta_reference: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> Generator[str, None, None]:
    """
    Unified entry point to read sequences from FASTQ, CRAM, BAM, SAM files, 1000G sample IDs, or URLs.
    """
    if source in ["HG00096", "HG00097", "HG00099", "HG00100"]:
        url = get_1000_genomes_url(source, file_type="fastq")
        yield from read_fastq_sequences(url, max_reads=max_reads, cache_dir=cache_dir)
        return

    lower_source = source.lower()
    if lower_source.endswith(".cram") or lower_source.endswith(".bam") or lower_source.endswith(".sam"):
        yield from read_cram_sequences(source, max_reads=max_reads, fasta_reference=fasta_reference)
    else:
        yield from read_fastq_sequences(source, max_reads=max_reads, cache_dir=cache_dir)


def get_kmers_from_sequence(sequence: str, k: int) -> List[str]:
    """Extract k-mers of length k from sequence, filtering out sequences with 'N'."""
    seq = sequence.upper()
    if 'N' in seq or len(seq) < k:
        return []
    return [seq[i : i + k] for i in range(len(seq) - k + 1) if 'N' not in seq[i : i + k]]
