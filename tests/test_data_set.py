from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from src.data_set import ESC50Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class DummyTransform:
    def __call__(self, audio_path: str | Path) -> torch.Tensor:
        return torch.ones(1, 128, 10, dtype=torch.float32)


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        pytest.skip(f"Config file not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def build_dataset(
    split: str | None = None,
    folds: list[int] | None = None,
    mode: str = "cnn",
    return_metadata: bool = False,
) -> ESC50Dataset:
    return ESC50Dataset(
        config_path=CONFIG_PATH,
        split=split,
        folds=folds,
        mode=mode,
        transform=DummyTransform(),
        return_metadata=return_metadata,
    )


def require_esc50_files(config: dict) -> None:
    metadata_path = PROJECT_ROOT / config["dataset"]["metadata_path"]
    audio_dir = PROJECT_ROOT / config["dataset"]["audio_dir"]

    if not metadata_path.exists():
        pytest.skip(f"ESC-50 metadata file not found: {metadata_path}")

    if not audio_dir.exists():
        pytest.skip(f"ESC-50 audio directory not found: {audio_dir}")


class TestESC50Dataset:
    def test_dataset_reads_metadata_from_config(self):
        config = load_config()
        require_esc50_files(config)

        dataset = build_dataset(split="train")

        assert isinstance(dataset.metadata, pd.DataFrame)
        assert len(dataset) > 0
        assert dataset.metadata_path == Path(config["dataset"]["metadata_path"])
        assert dataset.audio_dir == Path(config["dataset"]["audio_dir"])

    def test_dataset_filters_rows_by_train_split_folds(self):
        config = load_config()
        require_esc50_files(config)

        dataset = build_dataset(split="train")

        expected_folds = set(config["splits"]["train_folds"])
        actual_folds = set(dataset.metadata["fold"].unique().tolist())

        assert actual_folds == expected_folds

    def test_dataset_filters_rows_by_validation_split_folds(self):
        config = load_config()
        require_esc50_files(config)

        dataset = build_dataset(split="validation")

        expected_folds = set(config["splits"]["validation_folds"])
        actual_folds = set(dataset.metadata["fold"].unique().tolist())

        assert actual_folds == expected_folds

    def test_dataset_filters_rows_by_test_split_folds(self):
        config = load_config()
        require_esc50_files(config)

        dataset = build_dataset(split="test")

        expected_folds = set(config["splits"]["test_folds"])
        actual_folds = set(dataset.metadata["fold"].unique().tolist())

        assert actual_folds == expected_folds

    def test_custom_folds_override_split(self):
        config = load_config()
        require_esc50_files(config)

        dataset = build_dataset(split="train", folds=[5])

        actual_folds = set(dataset.metadata["fold"].unique().tolist())

        assert actual_folds == {5}

    def test_dataset_builds_full_audio_paths(self):
        config = load_config()
        require_esc50_files(config)

        dataset = build_dataset(split="train")

        assert len(dataset.audio_paths) == len(dataset)
        assert all(path.parent == dataset.audio_dir for path in dataset.audio_paths)
        assert all(path.suffix == ".wav" for path in dataset.audio_paths)

    def test_getitem_returns_features_and_label(self):
        config = load_config()
        require_esc50_files(config)

        dataset = build_dataset(split="train")

        features, label = dataset[0]

        assert isinstance(features, torch.Tensor)
        assert features.shape == (1, 128, 10)
        assert features.dtype == torch.float32
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long
        assert 0 <= label.item() < dataset.num_classes

    def test_getitem_can_return_metadata(self):
        config = load_config()
        require_esc50_files(config)

        dataset = build_dataset(split="train", return_metadata=True)

        features, label, metadata = dataset[0]

        assert isinstance(features, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert isinstance(metadata, dict)
        assert metadata["filename"].endswith(".wav")
        assert isinstance(metadata["category"], str)
        assert isinstance(metadata["fold"], int)
        assert isinstance(metadata["target"], int)
        assert isinstance(metadata["audio_path"], str)

    def test_dataset_uses_all_folds_when_split_and_folds_are_none(self):
        config = load_config()
        require_esc50_files(config)

        dataset = build_dataset(split=None, folds=None)

        actual_folds = set(dataset.metadata["fold"].unique().tolist())

        assert actual_folds == {1, 2, 3, 4, 5}

    def test_invalid_mode_raises_value_error(self):
        config = load_config()
        require_esc50_files(config)

        with pytest.raises(ValueError, match="Dataset mode is unsupported"):
            build_dataset(split="train", mode="invalid")

    def test_invalid_split_raises_value_error(self):
        config = load_config()
        require_esc50_files(config)

        with pytest.raises(ValueError, match="Unsupported split"):
            build_dataset(split="invalid")

    def test_missing_config_file_raises_file_not_found_error(self):
        missing_config_path = PROJECT_ROOT / "configs" / "missing_config.yaml"

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            ESC50Dataset(
                config_path=missing_config_path,
                split="train",
                transform=DummyTransform(),
            )