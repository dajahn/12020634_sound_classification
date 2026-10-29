import csv
from pathlib import Path

import pytest
import torch
import yaml

from src.data_loader import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        pytest.skip(f"Config file not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def get_first_audio_file(config: dict) -> Path:
    metadata_path = PROJECT_ROOT / config["dataset"]["metadata_path"]
    audio_dir = PROJECT_ROOT / config["dataset"]["audio_dir"]

    if not metadata_path.exists():
        pytest.skip(f"ESC-50 metadata file not found: {metadata_path}")

    if not audio_dir.exists():
        pytest.skip(f"ESC-50 audio directory not found: {audio_dir}")

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        reader = csv.DictReader(metadata_file)
        first_row = next(reader, None)

    if first_row is None:
        pytest.skip(f"ESC-50 metadata file is empty: {metadata_path}")

    audio_path = audio_dir / first_row["filename"]

    if not audio_path.exists():
        pytest.skip(f"ESC-50 audio file not found: {audio_path}")

    return audio_path


class TestDataLoader:
    def test_load_waveform_cnn_loads_existing_wav_file_correctly(self):
        config = load_config()
        audio_path = get_first_audio_file(config)

        sample_rate = config["audio"]["sample_rate_cnn"]
        duration_seconds = config["audio"]["duration_seconds"]
        expected_num_samples = int(sample_rate * duration_seconds)

        waveform = DataLoader.load_waveform_cnn(
            audio_path=audio_path,
            sample_rate=sample_rate,
            dur_seconds=duration_seconds,
        )

        assert isinstance(waveform, torch.Tensor)
        assert waveform.shape == (1, expected_num_samples)
        assert waveform.dtype == torch.float32
        assert not torch.isnan(waveform).any()
        assert not torch.isinf(waveform).any()

    def test_load_waveform_panns_loads_existing_wav_file_correctly(self):
        config = load_config()
        audio_path = get_first_audio_file(config)

        sample_rate = config["audio"]["sample_rate_panns"]
        duration_seconds = config["audio"]["duration_seconds"]
        expected_num_samples = int(sample_rate * duration_seconds)

        waveform = DataLoader.load_waveform_panns(
            audio_path=audio_path,
            sample_rate=sample_rate,
            dur_seconds=duration_seconds,
        )

        assert isinstance(waveform, torch.Tensor)
        assert waveform.shape == (1, expected_num_samples)
        assert waveform.dtype == torch.float32
        assert not torch.isnan(waveform).any()
        assert not torch.isinf(waveform).any()

    def test_load_waveform_loads_existing_wav_file_with_config_values(self):
        config = load_config()
        audio_path = get_first_audio_file(config)

        sample_rate = config["audio"]["sample_rate_cnn"]
        duration_seconds = config["audio"]["duration_seconds"]
        mono = config["audio"]["mono"]
        expected_channels = 1 if mono else None
        expected_num_samples = int(sample_rate * duration_seconds)

        waveform = DataLoader.load_waveform(
            audio_path=audio_path,
            target_sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            mono=mono,
        )

        assert isinstance(waveform, torch.Tensor)
        assert waveform.shape[-1] == expected_num_samples
        assert waveform.dtype == torch.float32
        assert not torch.isnan(waveform).any()
        assert not torch.isinf(waveform).any()

        if expected_channels is not None:
            assert waveform.shape[0] == expected_channels