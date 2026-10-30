import csv
from pathlib import Path

import pytest
import torch
import yaml

from src.preprocessing import LogMelPreprocessor, PannsPreprocessor


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


def build_log_mel_preprocessor(config: dict) -> LogMelPreprocessor:
    return LogMelPreprocessor(
        sample_rate=config["audio"]["sample_rate_cnn"],
        duration_seconds=config["audio"]["duration_seconds"],
        n_mels=config["feature"]["n_mels"],
        n_fft=config["feature"]["n_fft"],
        hop_len=config["feature"]["hop_length"],
    )


def build_panns_preprocessor(config: dict) -> PannsPreprocessor:
    return PannsPreprocessor(
        sample_rate=config["audio"]["sample_rate_panns"],
        duration_seconds=config["audio"]["duration_seconds"],
    )


class TestLogMelPreprocessor:
    def test_waveform_to_log_mel_returns_expected_shape_and_dtype(self):
        config = load_config()
        preprocessor = build_log_mel_preprocessor(config)

        sample_rate = config["audio"]["sample_rate_cnn"]
        duration_seconds = config["audio"]["duration_seconds"]
        n_mels = config["feature"]["n_mels"]

        num_samples = int(sample_rate * duration_seconds)
        waveform = torch.randn(1, num_samples)

        log_mel = preprocessor.waveform_to_log_mel(waveform)

        assert isinstance(log_mel, torch.Tensor)
        assert log_mel.ndim == 3
        assert log_mel.shape[0] == 1
        assert log_mel.shape[1] == n_mels
        assert log_mel.dtype == torch.float32
        assert not torch.isnan(log_mel).any()
        assert not torch.isinf(log_mel).any()

    def test_waveform_to_log_mel_converts_multi_channel_waveform_to_mono(self):
        config = load_config()
        preprocessor = build_log_mel_preprocessor(config)

        sample_rate = config["audio"]["sample_rate_cnn"]
        duration_seconds = config["audio"]["duration_seconds"]
        n_mels = config["feature"]["n_mels"]

        num_samples = int(sample_rate * duration_seconds)
        stereo_waveform = torch.randn(2, num_samples)

        log_mel = preprocessor.waveform_to_log_mel(stereo_waveform)

        assert isinstance(log_mel, torch.Tensor)
        assert log_mel.ndim == 3
        assert log_mel.shape[0] == 1
        assert log_mel.shape[1] == n_mels
        assert log_mel.dtype == torch.float32
        assert not torch.isnan(log_mel).any()
        assert not torch.isinf(log_mel).any()

    def test_waveform_to_log_mel_normalizes_output(self):
        config = load_config()
        preprocessor = build_log_mel_preprocessor(config)

        sample_rate = config["audio"]["sample_rate_cnn"]
        duration_seconds = config["audio"]["duration_seconds"]

        num_samples = int(sample_rate * duration_seconds)
        waveform = torch.randn(1, num_samples)

        log_mel = preprocessor.waveform_to_log_mel(waveform)

        assert torch.isclose(log_mel.mean(), torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(log_mel.std(), torch.tensor(1.0), atol=1e-4)

    def test_waveform_to_log_mel_raises_error_for_invalid_waveform_shape(self):
        preprocessor = LogMelPreprocessor()

        invalid_waveform = torch.randn(44100)

        with pytest.raises(ValueError, match="Expected waveform with shape"):
            preprocessor.waveform_to_log_mel(invalid_waveform)

    def test_call_loads_audio_and_returns_log_mel_spectrogram(self):
        config = load_config()
        audio_path = get_first_audio_file(config)
        preprocessor = build_log_mel_preprocessor(config)

        n_mels = config["feature"]["n_mels"]

        log_mel = preprocessor(audio_path)

        assert isinstance(log_mel, torch.Tensor)
        assert log_mel.ndim == 3
        assert log_mel.shape[0] == 1
        assert log_mel.shape[1] == n_mels
        assert log_mel.dtype == torch.float32
        assert not torch.isnan(log_mel).any()
        assert not torch.isinf(log_mel).any()


class TestPannsWaveformPreprocessor:
    def test_call_loads_audio_and_returns_expected_waveform_shape(self):
        config = load_config()
        audio_path = get_first_audio_file(config)
        preprocessor = build_panns_preprocessor(config)

        sample_rate = config["audio"]["sample_rate_panns"]
        duration_seconds = config["audio"]["duration_seconds"]
        expected_num_samples = int(sample_rate * duration_seconds)

        waveform = preprocessor(audio_path)

        assert isinstance(waveform, torch.Tensor)
        assert waveform.ndim == 1
        assert waveform.shape == (expected_num_samples,)
        assert waveform.dtype == torch.float32
        assert not torch.isnan(waveform).any()
        assert not torch.isinf(waveform).any()

    def test_panns_preprocessor_uses_32000_hz_default_sample_rate(self):
        preprocessor = PannsPreprocessor()

        assert preprocessor.sample_rate == 32000

    def test_panns_preprocessor_uses_5_seconds_default_duration(self):
        preprocessor = PannsPreprocessor()

        assert preprocessor.duration_seconds == 5.0

    def test_panns_output_is_raw_waveform_not_log_mel_spectrogram(self):
        config = load_config()
        audio_path = get_first_audio_file(config)
        preprocessor = build_panns_preprocessor(config)

        waveform = preprocessor(audio_path)

        assert waveform.ndim == 1