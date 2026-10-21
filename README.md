# kmer-karyotyper

`kmer-karyotyper` is a command-line tool and Python library for k-mer based karyotyping from FASTQ and CRAM sequence reads.

It supports:
- **Formats**: FASTQ (plain and gzipped `.fastq.gz`), BAM, SAM, and CRAM (`.cram`).
- **Karyotypes**: Standard sex karyotypes (XX, XY) as well as minor sex karyotypes ranging from Turner syndrome (X0) through trisomies (XXX, XXY, XYY), tetrasomies (XXXX, XXXY, XXYY, XYYY), and pentasomies (XXXXX, XXXXY, XXXYY, XXYYY, XYYYY).
- **Data Sources**: Local files, HTTP/HTTPS URLs, and built-in 1000 Genomes sample identifiers (e.g. `HG00096`, `HG00097`, `HG00099`, `HG00100`).
- **Commands**: CLI scripts for training custom k-mer classifier models (`kmer-train` / `kmer-karyotyper train`) and classifying unknown samples (`kmer-classify` / `kmer-karyotyper classify`).

---

## Installation

This project is managed with [Poetry](https://python-poetry.org/).

To install the dependencies and the CLI tools:

```bash
poetry install
```

---

## CLI Usage

### 1. Training a Model (`kmer-train` or `kmer-karyotyper train`)

Train a classifier by providing labelled samples in the format `<source>:<label>`. Sources can be local FASTQ/CRAM paths, URLs, or 1000 Genomes identifiers.

```bash
# Training with local FASTQ and CRAM files including minor sex karyotypes
poetry run kmer-train \
  --samples sample_XX.fastq:XX sample_XY.cram:XY sample_X0.fastq:X0 sample_XXY.fastq:XXY sample_XXXXX.fastq:XXXXX \
  --output-model model.pt \
  --k-size 31 \
  --max-vocab 1000 \
  --epochs 30

# Training using 1000 Genomes samples
poetry run kmer-train \
  --samples HG00096:XX HG00097:XY \
  --output-model 1000g_model.pt \
  --max-reads 50000
```

#### Training Options
- `--samples`: One or more sample sources with labels, e.g. `path/to/file.fastq:XX` or `HG00096:0`.
- `--output-model`: Output path for saved model checkpoint (default: `kmer_karyotype_model.pt`).
- `--k-size`: Size $k$ of k-mers to extract (default: `31`).
- `--max-vocab`: Maximum number of unique k-mer features to use (default: `1000`).
- `--epochs`: Number of training epochs (default: `30`).
- `--lr`: Learning rate (default: `0.005`).
- `--max-reads`: Optional limit on reads parsed per sample for faster processing.
- `--fasta-ref`: Optional FASTA reference file if required for CRAM decompression.

---

### 2. Classifying Samples (`kmer-classify` or `kmer-karyotyper classify`)

Classify sample karyotypes using a previously trained model checkpoint.

```bash
# Classify local FASTQ / CRAM files
poetry run kmer-classify \
  --model model.pt \
  --samples test_sample1.fastq test_sample2.cram

# Classify 1000 Genomes sample
poetry run kmer-classify \
  --model 1000g_model.pt \
  --samples HG00099 HG00100
```

#### Classification Options
- `--model`: Path to trained PyTorch model checkpoint (`.pt`).
- `--samples`: One or more sample paths, URLs, or 1000 Genomes IDs to classify.
- `--max-reads`: Optional limit on reads parsed per sample.
- `--fasta-ref`: Optional FASTA reference file for CRAM files.

---

## Running Tests

Run unit and integration tests using `pytest`:

```bash
poetry run pytest
```
