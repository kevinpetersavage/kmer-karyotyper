"""
Command line interface for kmer-karyotyper training and classification.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from kmer_karyotyper.data import is_url, get_1000_genomes_url
from kmer_karyotyper.model import (
    KmerClassifier,
    KmerKaryotypeDataset,
    build_kmer_vocabulary,
    load_model,
    save_model
)


DEFAULT_LABEL_MAP = {0: "XX (Female)", 1: "XY (Male)"}
REV_LABEL_MAP = {"XX": 0, "XX (Female)": 0, "0": 0, "XY": 1, "XY (Male)": 1, "1": 1}


def parse_sample_arg(sample_str: str) -> Tuple[str, Optional[int]]:
    """
    Parse sample input string which may be 'path_or_id:label' or just 'path_or_id'.
    Handles URLs correctly (e.g., 'https://example.com/file.fastq.gz:XX').
    Labels can be numeric (0, 1) or string ('XX', 'XY').
    """
    if ":" in sample_str:
        parts = sample_str.rsplit(":", 1)
        possible_label = parts[1].strip()

        if possible_label in REV_LABEL_MAP:
            return parts[0], REV_LABEL_MAP[possible_label]
        elif possible_label.isdigit():
            return parts[0], int(possible_label)

    return sample_str, None


def train_cli(args_list: Optional[List[str]] = None) -> None:
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

    num_classes = len(set(labels))
    if num_classes < 2:
        num_classes = 2

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


def classify_cli(args_list: Optional[List[str]] = None) -> None:
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

    train_parser = subparsers.add_parser("train", help="Train a k-mer karyotype classifier")
    classify_parser = subparsers.add_parser("classify", help="Classify samples with a trained model")

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_cli(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "classify":
        classify_cli(sys.argv[2:])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
