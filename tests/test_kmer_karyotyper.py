"""
Tests for kmer-karyotyper data loading, modeling, CLI, FASTQ, and CRAM processing.
"""

import gzip
import os
import pytest
import pysam

from kmer_karyotyper.data import (
    get_kmers_from_sequence,
    read_fastq_sequences,
    read_cram_sequences,
    read_sequences,
    get_1000_genomes_url
)
from kmer_karyotyper.model import (
    KmerClassifier,
    KmerKaryotypeDataset,
    build_kmer_vocabulary,
    save_model,
    load_model
)
from kmer_karyotyper.cli import train_cli, classify_cli, parse_sample_arg


@pytest.fixture
def sample_fastq_files(tmp_path):
    # Sample 1: XX (female)
    fq1 = tmp_path / "sample_XX.fastq"
    with open(fq1, "w") as f:
        f.write("@READ1\nACACACACACACACACACACACACACACAC\n+\n" + "~" * 30 + "\n")
        f.write("@READ2\nTGTGTGACACACACACACACACACACACAC\n+\n" + "~" * 30 + "\n")

    # Sample 2: XY (male)
    fq2 = tmp_path / "sample_XY.fastq.gz"
    with gzip.open(fq2, "wt") as f:
        f.write("@READ1\nGGGGGGCCCCCCGGGGGGCCCCCCGGGGGG\n+\n" + "~" * 30 + "\n")
        f.write("@READ2\nCCCCCCGGGGGGCCCCCCGGGGGGCCCCCC\n+\n" + "~" * 30 + "\n")

    return str(fq1), str(fq2)


@pytest.fixture
def sample_sam_file(tmp_path):
    sam_path = tmp_path / "sample.sam"
    header = {"HD": {"VN": "1.0"}, "SQ": [{"LN": 1000, "SN": "chr1"}]}
    with pysam.AlignmentFile(sam_path, "wh", header=header) as out_sam:
        a = pysam.AlignedSegment()
        a.query_name = "read1"
        a.query_sequence = "ATCGATCGATCGATCGATCGATCG"
        a.flag = 0
        a.reference_id = 0
        a.reference_start = 100
        a.mapping_quality = 20
        a.cigar = ((0, 24),)
        a.query_qualities = pysam.qualitystring_to_array("~" * 24)
        out_sam.write(a)
    return str(sam_path)


def test_get_kmers_from_sequence():
    seq = "ACGTACGT"
    kmers = get_kmers_from_sequence(seq, 3)
    assert kmers == ["ACG", "CGT", "GTA", "TAC", "ACG", "CGT"]

    # Test 'N' handling
    seq_n = "ACGNACGT"
    assert get_kmers_from_sequence(seq_n, 3) == []


def test_read_fastq_sequences(sample_fastq_files):
    fq1, fq2 = sample_fastq_files
    seqs1 = list(read_fastq_sequences(fq1))
    assert len(seqs1) == 2
    assert seqs1[0] == "ACACACACACACACACACACACACACACAC"

    seqs2 = list(read_fastq_sequences(fq2))
    assert len(seqs2) == 2
    assert seqs2[0] == "GGGGGGCCCCCCGGGGGGCCCCCCGGGGGG"


def test_read_cram_and_sam_sequences(sample_sam_file):
    seqs = list(read_cram_sequences(sample_sam_file))
    assert len(seqs) == 1
    assert seqs[0] == "ATCGATCGATCGATCGATCGATCG"

    # Unified reader
    seqs_unified = list(read_sequences(sample_sam_file))
    assert seqs_unified == ["ATCGATCGATCGATCGATCGATCG"]


def test_1000_genomes_url_resolution():
    url_96 = get_1000_genomes_url("HG00096", file_type="fastq")
    assert "SRR062634" in url_96

    url_96_cram = get_1000_genomes_url("HG00096", file_type="cram")
    assert "HG00096.final.cram" in url_96_cram

    with pytest.raises(ValueError):
        get_1000_genomes_url("UNKNOWN_SAMPLE")


def test_parse_sample_arg():
    path, label = parse_sample_arg("path/to/sample.fastq:XX")
    assert path == "path/to/sample.fastq"
    assert label == 0

    path, label = parse_sample_arg("path/to/sample.cram:1")
    assert path == "path/to/sample.cram"
    assert label == 1

    path, label = parse_sample_arg("HG00096:XY")
    assert path == "HG00096"
    assert label == 1

    # Minor sex karyotypes testing
    path, label = parse_sample_arg("sample_turner.fastq:X0")
    assert path == "sample_turner.fastq"
    assert label == 2

    path, label = parse_sample_arg("sample_turner.fastq:Turner")
    assert path == "sample_turner.fastq"
    assert label == 2

    path, label = parse_sample_arg("sample_klinefelter.fastq:XXY")
    assert path == "sample_klinefelter.fastq"
    assert label == 4

    path, label = parse_sample_arg("sample_pentasomy_x.fastq:XXXXX")
    assert path == "sample_pentasomy_x.fastq"
    assert label == 10

    path, label = parse_sample_arg("sample_pentasomy_49.fastq:XXXXY")
    assert path == "sample_pentasomy_49.fastq"
    assert label == 11


def test_cli_train_and_classify_minor_karyotypes(tmp_path, capsys):
    # Create FASTQ files for multiple minor karyotypes: XX, XY, X0 (Turner), XXY (Klinefelter), XXXXX (Pentasomy X)
    samples = {
        "sample_XX.fastq": ("XX", "ACACACACACACACACACACACACACACAC"),
        "sample_XY.fastq": ("XY", "GGGGGGCCCCCCGGGGGGCCCCCCGGGGGG"),
        "sample_X0.fastq": ("X0", "AAAAAAAAAAAAAAACCCCCCCCCCCCCCC"),
        "sample_XXY.fastq": ("XXY", "TTTTTTTTTTTTTTTGGGGGGGGGGGGGGG"),
        "sample_XXXXX.fastq": ("XXXXX", "ATATATATATATATATATATATATATATAT"),
    }

    sample_args = []
    for fname, (label_str, seq) in samples.items():
        fpath = str(tmp_path / fname)
        with open(fpath, "w") as f:
            f.write(f"@READ1\n{seq}\n+\n" + "~" * len(seq) + "\n")
        sample_args.append(f"{fpath}:{label_str}")

    model_path = str(tmp_path / "minor_karyotypes_model.pt")

    train_args = [
        "--samples", *sample_args,
        "--output-model", model_path,
        "--k-size", "5",
        "--epochs", "10",
        "--max-vocab", "100"
    ]
    train_cli(train_args)
    assert os.path.exists(model_path)

    # Classify all samples
    test_files = [item.split(":")[0] for item in sample_args]
    classify_args = [
        "--model", model_path,
        "--samples", *test_files
    ]
    classify_cli(classify_args)

    captured = capsys.readouterr()
    assert "Classification Results" in captured.out
    assert "Predicted Karyotype" in captured.out


def test_model_training_and_serialization(sample_fastq_files, tmp_path):
    fq1, fq2 = sample_fastq_files
    filepaths = [fq1, fq2]
    labels = [0, 1]
    k_size = 5

    vocab = build_kmer_vocabulary(filepaths, k=k_size, max_vocab_size=100)
    assert len(vocab) > 0

    dataset = KmerKaryotypeDataset(filepaths, labels, vocab, k=k_size)
    assert len(dataset) == 2

    feat, lbl = dataset[0]
    assert feat.shape[0] == len(vocab)
    assert lbl.item() == 0

    model = KmerClassifier(vocab_size=len(vocab), num_classes=2)
    model.train()
    out = model(feat.unsqueeze(0))
    assert out.shape == (1, 2)

    model_path = str(tmp_path / "model.pt")
    label_map = {0: "XX", 1: "XY"}
    save_model(model, vocab, k_size, label_map, model_path)
    assert os.path.exists(model_path)

    loaded_model, loaded_vocab, loaded_k, loaded_map = load_model(model_path)
    assert loaded_k == k_size
    assert loaded_vocab == vocab
    assert loaded_map == label_map


def test_cli_train_and_classify_e2e(sample_fastq_files, tmp_path, capsys):
    fq1, fq2 = sample_fastq_files
    model_path = str(tmp_path / "e2e_model.pt")

    # Run train CLI
    train_args = [
        "--samples", f"{fq1}:XX", f"{fq2}:XY",
        "--output-model", model_path,
        "--k-size", "5",
        "--epochs", "5",
        "--max-vocab", "50"
    ]
    train_cli(train_args)
    assert os.path.exists(model_path)

    # Run classify CLI
    classify_args = [
        "--model", model_path,
        "--samples", fq1, fq2
    ]
    classify_cli(classify_args)

    captured = capsys.readouterr()
    assert "Classification Results" in captured.out
    assert "Predicted Karyotype" in captured.out
