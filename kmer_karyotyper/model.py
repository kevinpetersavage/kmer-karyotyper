"""
PyTorch dataset, classifier model architecture, and serialization routines.
"""

import collections
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from kmer_karyotyper.data import get_kmers_from_sequence, read_sequences


class KmerKaryotypeDataset(Dataset):
    """
    Dataset that converts reads from FASTQ/CRAM files or URLs into k-mer count feature vectors.
    """

    def __init__(
        self,
        filepaths: List[str],
        labels: Optional[List[int]],
        kmer_to_idx: Dict[str, int],
        k: int,
        max_reads_per_sample: Optional[int] = None,
        fasta_reference: Optional[str] = None
    ):
        self.filepaths = filepaths
        self.labels = labels
        self.kmer_to_idx = kmer_to_idx
        self.k = k
        self.vocab_size = len(kmer_to_idx)
        self.max_reads_per_sample = max_reads_per_sample
        self.fasta_reference = fasta_reference

    def __len__(self) -> int:
        return len(self.filepaths)

    def extract_features(self, filepath: str) -> torch.Tensor:
        counts = collections.Counter()
        sequences = read_sequences(
            filepath,
            max_reads=self.max_reads_per_sample,
            fasta_reference=self.fasta_reference
        )
        for seq in sequences:
            kmers = get_kmers_from_sequence(seq, self.k)
            counts.update(kmers)

        feature_vector = torch.zeros(self.vocab_size, dtype=torch.float32)
        for kmer, count in counts.items():
            if kmer in self.kmer_to_idx:
                feature_vector[self.kmer_to_idx[kmer]] = count

        return feature_vector

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[torch.Tensor, str]]:
        filepath = self.filepaths[idx]
        feature_vector = self.extract_features(filepath)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return feature_vector, label
        else:
            return feature_vector, filepath


class KmerClassifier(nn.Module):
    """
    Neural network classifier for k-mer feature vectors.
    """

    def __init__(self, vocab_size: int, num_classes: int = 2, hidden_dim: int = 32, dropout: float = 0.2):
        super(KmerClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.network = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize feature vector to prevent scaling issues with high counts
        x_norm = nn.functional.normalize(x, p=2, dim=1 if x.dim() > 1 else 0)
        return self.network(x_norm)


def build_kmer_vocabulary(
    filepaths: List[str],
    k: int,
    max_vocab_size: Optional[int] = 1000,
    max_reads_per_sample: Optional[int] = None,
    fasta_reference: Optional[str] = None
) -> Dict[str, int]:
    """
    Build a k-mer vocabulary mapping (kmer -> idx) from a list of sample files/URLs.
    Optionally select top max_vocab_size most common k-mers across all samples.
    """
    all_kmer_counts = collections.Counter()
    for filepath in filepaths:
        for seq in read_sequences(filepath, max_reads=max_reads_per_sample, fasta_reference=fasta_reference):
            kmers = get_kmers_from_sequence(seq, k)
            all_kmer_counts.update(kmers)

    if max_vocab_size and len(all_kmer_counts) > max_vocab_size:
        most_common = [kmer for kmer, _ in all_kmer_counts.most_common(max_vocab_size)]
    else:
        most_common = sorted(list(all_kmer_counts.keys()))

    return {kmer: idx for idx, kmer in enumerate(most_common)}


def save_model(
    model: KmerClassifier,
    kmer_to_idx: Dict[str, int],
    k: int,
    label_map: Dict[int, str],
    output_path: str
) -> None:
    """
    Save trained model state, vocabulary, k-mer size, and label mapping to a file.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab_size": model.vocab_size,
        "num_classes": model.num_classes,
        "hidden_dim": model.hidden_dim,
        "kmer_to_idx": kmer_to_idx,
        "k": k,
        "label_map": label_map
    }
    torch.save(checkpoint, output_path)


def load_model(model_path: str) -> Tuple[KmerClassifier, Dict[str, int], int, Dict[int, str]]:
    """
    Load saved model state, vocabulary, k-mer size, and label mapping.
    """
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    model = KmerClassifier(
        vocab_size=checkpoint["vocab_size"],
        num_classes=checkpoint["num_classes"],
        hidden_dim=checkpoint["hidden_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    kmer_to_idx = checkpoint["kmer_to_idx"]
    k = checkpoint["k"]
    label_map = checkpoint["label_map"]

    return model, kmer_to_idx, k, label_map
