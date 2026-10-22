"""
Command line interface for kmer-karyotyper training and classification.
"""

import argparse
import sys
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from kmer_karyotyper.model import (
    KmerClassifier,
    KmerKaryotypeDataset,
    build_kmer_vocabulary,
    load_model,
    save_model
)


DEFAULT_LABEL_MAP = {
    0: "XX (Female)",
    1: "XY (Male)",
    2: "X0 (Turner syndrome)",
    3: "XXX (Triple X syndrome)",
    4: "XXY (Klinefelter syndrome)",
    5: "XYY (Jacobs syndrome)",
    6: "XXXX (Tetrasomy X)",
    7: "XXXY (48, XXXY syndrome)",
    8: "XXYY (48, XXYY syndrome)",
    9: "XYYY (48, XYYY syndrome)",
    10: "XXXXX (Pentasomy X)",
    11: "XXXXY (49, XXXXY syndrome)",
    12: "XXXYY (49, XXXYY syndrome)",
    13: "XXYYY (49, XXYYY syndrome)",
    14: "XYYYY (49, XYYYY syndrome)",
    15: "YY",
}

REV_LABEL_MAP = {
    # 0: XX
    "XX": 0, "XX (FEMALE)": 0, "FEMALE": 0, "0": 0,
    # 1: XY
    "XY": 1, "XY (MALE)": 1, "MALE": 1, "1": 1,
    # 2: Turner / X0
    "X0": 2, "XO": 2, "X": 2, "TURNER": 2, "TURNERS": 2, "X0 (TURNER SYNDROME)": 2, "2": 2,
    # 3: Triple X / XXX
    "XXX": 3, "XXX (TRIPLE X SYNDROME)": 3, "TRIPLE X": 3, "3": 3,
    # 4: Klinefelter / XXY
    "XXY": 4, "XXY (KLINEFELTER SYNDROME)": 4, "KLINEFELTER": 4, "4": 4,
    # 5: Jacobs / XYY
    "XYY": 5, "XYY (JACOBS SYNDROME)": 5, "JACOBS": 5, "5": 5,
    # 6: Tetrasomy X / XXXX
    "XXXX": 6, "XXXX (TETRASOMY X)": 6, "TETRASOMY X": 6, "6": 6,
    # 7: XXXY
    "XXXY": 7, "XXXY (48, XXXY SYNDROME)": 7, "7": 7,
    # 8: XXYY
    "XXYY": 8, "XXYY (48, XXYY SYNDROME)": 8, "8": 8,
    # 9: XYYY
    "XYYY": 9, "XYYY (48, XYYY SYNDROME)": 9, "9": 9,
    # 10: Pentasomy X / XXXXX
    "XXXXX": 10, "XXXXX (PENTASOMY X)": 10, "PENTASOMY X": 10, "10": 10,
    # 11: XXXXY
    "XXXXY": 11, "XXXXY (49, XXXXY SYNDROME)": 11, "11": 11,
    # 12: XXXYY
    "XXXYY": 12, "XXXYY (49, XXXYY SYNDROME)": 12, "12": 12,
    # 13: XXYYY
    "XXYYY": 13, "XXYYY (49, XXYYY SYNDROME)": 13, "13": 13,
    # 14: XYYYY
    "XYYYY": 14, "XYYYY (49, XYYYY SYNDROME)": 14, "14": 14,
    # 15: YY
    "YY": 15, "15": 15,
}


def canonicalize_karyotype(label_str: str) -> Optional[str]:
    """
    Canonicalize a karyotype string into standard form (e.g. 'x0' -> 'X0', 'yxxxx' -> 'XXXXY').
    Supports sex chromosome karyotypes from monosomy X (Turner) up to pentasomies.
    """
    clean = label_str.strip().upper()
    if clean in REV_LABEL_MAP:
        idx = REV_LABEL_MAP[clean]
        for k, v in REV_LABEL_MAP.items():
            if v == idx and k.isalpha() and len(k) <= 5:
                return k

    if clean in ("X", "X0", "XO", "TURNER", "TURNERS"):
        return "X0"

    if all(c in ("X", "Y", "0", "O") for c in clean):
        x_count = clean.count("X")
        y_count = clean.count("Y")
        if x_count == 1 and y_count == 0 and ("0" in clean or "O" in clean or len(clean) == 1):
            return "X0"
        if x_count + y_count > 0:
            return "X" * x_count + "Y" * y_count

    return clean


def parse_sample_arg(sample_str: str) -> Tuple[str, Optional[Union[int, str]]]:
    """
    Parse sample input string which may be 'path_or_id:label' or just 'path_or_id'.
    Handles URLs correctly (e.g., 'https://example.com/file.fastq.gz:XX').
    Labels can be numeric, standard karyotype names ('XX', 'XY', 'X0', 'XXXXX', etc.), or custom strings.
    """
    if ":" in sample_str:
        parts = sample_str.rsplit(":", 1)
        possible_label = parts[1].strip()
        upper_label = possible_label.upper()

        if upper_label in REV_LABEL_MAP:
            return parts[0], REV_LABEL_MAP[upper_label]
        if possible_label.isdigit():
            return parts[0], int(possible_label)

        canon = canonicalize_karyotype(possible_label)
        if canon and canon in REV_LABEL_MAP:
            return parts[0], REV_LABEL_MAP[canon]
        return parts[0], possible_label

    return sample_str, None


def _build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a k-mer karyotyping model using FASTQ or CRAM sequence data."
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        required=True,
        help="List of sample paths, URLs, or 1000 Genomes sample IDs formatted as source:label (e.g. sample.fastq:XX HG00096:0 sample.cram:XY)."
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="kmer_karyotype_model.pt",
        help="Path to save trained model checkpoint (.pt)."
    )
    parser.add_argument(
        "--k-size",
        type=int,
        default=31,
        help="K-mer size (default: 31)."
    )
    parser.add_argument(
        "--max-vocab",
        type=int,
        default=1000,
        help="Maximum vocabulary size for k-mer features (default: 1000)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
        help="Learning rate (default: 0.005)."
    )
    parser.add_argument(
        "--max-reads",
        type=int,
        default=None,
        help="Maximum reads to parse per sample (for fast demo/training)."
    )
    parser.add_argument(
        "--fasta-ref",
        type=str,
        default=None,
        help="Optional FASTA reference for CRAM files."
    )
    return parser


def train_cli(args_list: Optional[List[str]] = None) -> None:
    parser = _build_train_parser()
    args = parser.parse_args(args_list)

    filepaths = []
    labels = []

    for item in args.samples:
        src, lbl = parse_sample_arg(item)
        if lbl is None:
            raise ValueError(f"Sample '{item}' must specify a valid label (e.g. {item}:XX or {item}:0).")
        filepaths.append(src)
        labels.append(lbl)

    print(f"Building k-mer vocabulary (k={args.k_size}, max_vocab={args.max_vocab})...")
    kmer_to_idx = build_kmer_vocabulary(
        filepaths,
        k=args.k_size,
        max_vocab_size=args.max_vocab,
        max_reads_per_sample=args.max_reads,
        fasta_reference=args.fasta_ref
    )
    print(f"Vocabulary created with {len(kmer_to_idx)} unique {args.k_size}-mers.")

    dataset = KmerKaryotypeDataset(
        filepaths=filepaths,
        labels=labels,
        kmer_to_idx=kmer_to_idx,
        k=args.k_size,
        max_reads_per_sample=args.max_reads,
        fasta_reference=args.fasta_ref
    )

    batch_size = min(4, len(filepaths))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    max_label_val = max(labels) if labels else 0
    num_classes = max(2, max_label_val + 1)

    model = KmerClassifier(vocab_size=len(kmer_to_idx), num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    print("Training model...")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for features, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % max(1, args.epochs // 5) == 0 or epoch == args.epochs - 1:
            print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {epoch_loss:.4f}")

    label_map = DEFAULT_LABEL_MAP
    save_model(model, kmer_to_idx, args.k_size, label_map, args.output_model)
    print(f"Model saved successfully to {args.output_model}")


def _build_classify_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify karyotype from FASTQ or CRAM sequence data using a trained model."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt)."
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        required=True,
        help="List of sample paths, URLs, or 1000 Genomes sample IDs to classify."
    )
    parser.add_argument(
        "--max-reads",
        type=int,
        default=None,
        help="Maximum reads to parse per sample."
    )
    parser.add_argument(
        "--fasta-ref",
        type=str,
        default=None,
        help="Optional FASTA reference for CRAM files."
    )
    return parser


def classify_cli(args_list: Optional[List[str]] = None) -> None:
    parser = _build_classify_parser()
    args = parser.parse_args(args_list)

    model, kmer_to_idx, k_size, label_map = load_model(args.model)

    filepaths = [parse_sample_arg(item)[0] for item in args.samples]

    dataset = KmerKaryotypeDataset(
        filepaths=filepaths,
        labels=None,
        kmer_to_idx=kmer_to_idx,
        k=k_size,
        max_reads_per_sample=args.max_reads,
        fasta_reference=args.fasta_ref
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("\n--- Classification Results ---")
    model.eval()
    with torch.no_grad():
        for features, paths in dataloader:
            outputs = model(features)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_idx].item()
            pred_label = label_map.get(predicted_idx, str(predicted_idx))

            sample_path = paths[0]
            print(f"Sample: {sample_path} | Predicted Karyotype: {pred_label} (Confidence: {confidence:.2%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="kmer-karyotyper CLI tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Train a k-mer karyotype classifier")
    subparsers.add_parser("classify", help="Classify samples with a trained model")

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_cli(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "classify":
        classify_cli(sys.argv[2:])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
