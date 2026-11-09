from fileinput import filename
from pathlib import Path
from typing import Callable, Literal

import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

from src.preprocessing import LogMelPreprocessor, PannsPreprocessor

DatasetMode = Literal["cnn", "panns"]
SplitName = Literal["train", "validation", "test"]

class ESC50Dataset(Dataset):
    """
       PyTorch Dataset for the ESC-50 environmental sound classification dataset.

       The dataset:
       - reads ESC-50 metadata
       - filters samples by selected folds
       - builds full paths to audio files
       - returns processed audio features and numeric labels
       - can optionally return metadata

       Supported modes:
       - "cnn": returns normalized log-mel spectrograms using LogMelPreprocessor
       - "panns": returns fixed-length waveform tensors using PannsPreprocessor
       """

    def __init__(self,
                 config_path: str | Path ="configs/config.yaml",
                split: SplitName|None = None,
                 folds: list[int] | tuple[int, ...] | None = None,
                 mode: DatasetMode = "cnn",
                 transform : Callable | None = None,
                 return_metadata: bool = False):
        """
               Args:
                   config_path:
                       Path to the project YAML configuration file.
                   split:
                       Dataset split to use. Supported values are "train", "validation", and "test".
                       If provided, folds are read from config.yaml.
                   folds:
                       Custom selected ESC-50 folds. If provided, this overrides split.
                       If both split and folds are None, all folds are used.
                   mode:
                       Dataset mode. Use "cnn" for log-mel spectrograms or "panns" for waveforms.
                   transform:
                       Optional custom transform/preprocessor. If provided, it overrides the default
                       preprocessor selected by mode.
                   return_metadata:
                       If True, __getitem__ returns metadata together with features and label.
        """
        config_path = Path(config_path)
        self.config_path = config_path
        self.config = self._load_config(self.config_path)
        dataset_config = self.config["dataset"]
        audio_config = self.config["audio"]
        feature_config = self.config["feature"]

        self.metadata_path = Path(dataset_config["metadata_path"])
        self.audio_dir = Path(dataset_config["audio_dir"])
        self.num_classes = int(dataset_config["num_classes"])
        self.mode = mode
        self.return_metadata = return_metadata

        if mode not in ("cnn", "panns"):
            raise ValueError(f"Dataset mode is unsupported: {mode} does not equal 'cnn' or 'panns'.")
        if folds is not None:
            self.folds = list(folds)
        elif split is not None:
            self.folds = self._get_folds_for_split(split)
        else:
            self.folds = None

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        if not self.audio_dir.exists():
            raise  FileNotFoundError(f"Audio directory not found: {self.audio_dir}")

        self.metadata = pd.read_csv(self.metadata_path)

        required_columns = {"filename", "fold", "target", "category"}
        missing_columns = required_columns.difference(self.metadata.columns)
        if missing_columns:
            raise ValueError(f"Metadata file is missing columns: {sorted(missing_columns)}")

        if self.folds is not None:
            self.metadata = self.metadata[self.metadata["fold"].isin(self.folds)]
        self.metadata = self.metadata.reset_index(drop = True)

        self.audio_paths =[self.audio_dir/fn for fn in self.metadata["filename"].tolist()]

        if transform is not None:
            self.transform = transform
        elif mode == "cnn":
            self.transform = LogMelPreprocessor(
                sample_rate=int(audio_config["sample_rate_cnn"]),
                duration_seconds=float(audio_config["duration_seconds"]),
                n_mels=int(feature_config["n_mels"]),
                n_fft=int(feature_config["n_fft"]),
                hop_len=int(feature_config["hop_length"])
            )
        else:
            self.transform = PannsPreprocessor(
                sample_rate=int(audio_config["sample_rate_panns"]),
                duration_seconds=float(audio_config["duration_seconds"])
            )


    @staticmethod
    def _load_config(config_path: Path) -> dict:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if config is None:
            raise ValueError(f"Config file is empty: {config_path}")

        required_sections = {"dataset", "audio", "feature", "splits"}
        missing_sections = required_sections.difference(config.keys())
        if missing_sections:
            raise ValueError(f"Config file is missing required sections: {missing_sections}")

        return config

    def __len__(self) -> int:
        return len(self.metadata)

    def _get_folds_for_split(self, split: SplitName) -> list[int]:
        split_to_config_keys = {
            "train": "train_folds",
            "validation": "validation_folds",
            "test": "test_folds"
        }
        if split not in split_to_config_keys:
            raise ValueError(f"Unsupported split: {split}")
        config_key = split_to_config_keys[split]
        return list(self.config["splits"][config_key])

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict]:
        row = self.metadata.iloc[index]
        audio_path = self.audio_paths[index]

        features = self.transform(audio_path)
        label = torch.tensor(int(row["target"]), dtype=torch.long)

        if not self.return_metadata:
            return  features, label

        metadata = {
            "filename": row["filename"],
            "category": row["category"],
            "fold": int(row["fold"]),
            "target": int(row["target"]),
            "audio_path": str(audio_path)
        }
        return features, label, metadata

class PannsEmbeddingDataset(Dataset):
    """
    Dataset for precomputed PANNs embeddings.

    Expected file format:
        {
            "embeddings": Tensor[num_examples, embedding_dim],
            "labels": Tensor[num_examples]
        }

    These files are produced by running: python -m src.training.precompute_panns_embeddings
    """

    def __init__(self, embeddings_path: str | Path) -> None:
        embeddings_path = Path(embeddings_path)

        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"PANNs embedding file not found: {embeddings_path}. "
                "Run `python -m src.training.precompute_panns_embeddings` first."
            )

        data = torch.load(
            embeddings_path,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(data, dict):
            raise TypeError(
                f"Expected a dictionary in {embeddings_path}, got {type(data).__name__}"
            )

        if "embeddings" not in data or "labels" not in data:
            raise ValueError(
                f"Embedding file {embeddings_path} must contain 'embeddings' and 'labels'."
            )

        self.embeddings = data["embeddings"].to(torch.float32)
        self.labels = data["labels"].to(torch.long)

        if self.embeddings.ndim != 2:
            raise ValueError(
                f"Expected embeddings with shape [num_examples, embedding_dim], "
                f"got {tuple(self.embeddings.shape)}"
            )

        if self.labels.ndim != 1:
            raise ValueError(
                f"Expected labels with shape [num_examples], got {tuple(self.labels.shape)}"
            )

        if self.embeddings.shape[0] != self.labels.shape[0]:
            raise ValueError(
                "Number of embeddings does not match number of labels: "
                f"{self.embeddings.shape[0]} != {self.labels.shape[0]}"
            )

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[index], self.labels[index]
