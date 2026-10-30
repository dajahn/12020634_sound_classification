from pathlib import Path

import torch
import torchaudio


from src.data_loader import DataLoader


class LogMelPreprocessor:
    """
    Convert audio waveforms into normalized log-mel spectrograms for the compact CNN baseline.

    Pipeline:
    1. Load waveform.
    2. Convert to mono if needed.
    3. Resample if needed.
    4. Pad or trim to fixed length if needed.
    5. Compute mel spectrogram.
    6. Convert to logarithmic scale.
    7. Normalize.

    Output shape:
        [1, n_mels, time_frames]
    """

    def __init__(self,
                sample_rate: int = 44100,
                duration_seconds: float = 5.0,
                n_mels: int = 128,
                n_fft: int = 2048,
                hop_len: int = 512,
                eps: float = 1e-10) -> None:
        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.eps = eps

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate = sample_rate,
            n_fft = n_fft,
            hop_length=hop_len,
            n_mels= n_mels,
            power = 2.0
        )

    def load_waveform(self, audio_path: str | Path) -> torch.Tensor:
        return DataLoader.load_waveform_cnn(audio_path=audio_path, sample_rate=self.sample_rate, dur_seconds=self.duration_seconds)

    def waveform_to_log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """
                Convert a fixed-length waveform into a normalized log-mel spectrogram.
                Args:
                    waveform:
                        Tensor with shape [1, num_samples].

                Returns:
                    Normalized log-mel spectrogram with shape [1, n_mels, time_frames].
                """
        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform with shape [channels, samples], got {tuple(waveform.shape)}")
        if waveform.shape[0] != 1: #convert to mono
            waveform = waveform.mean(dim=0, keepdim=True)

        mel_spec = self.mel_spectrogram(waveform)
        log_mel_spec = torch.log(mel_spec + self.eps)

        mean = log_mel_spec.mean()
        std = log_mel_spec.std()

        if std <= self.eps:
            normalized_log_mel = log_mel_spec - mean
        else:
            normalized_log_mel = (log_mel_spec - mean) / std
        return normalized_log_mel.to(torch.float32)

    def __call__(self, audio_path: str | Path) -> torch.Tensor:
        """
                Load an audio file and return a normalized log-mel spectrogram.
                Args:
                    audio_path:
                        Path to an audio file.

                Returns:
                    Tensor with shape [1, n_mels, time_frames].
                """
        waveform = self.load_waveform(audio_path)
        return self.waveform_to_log_mel(waveform)

class PannsPreprocessor:
    def __init__(self, sample_rate = 32000, duration_seconds =5.0):
        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds

    def __call__(self, audio_path: str | Path) -> torch.Tensor:
        waveform = DataLoader.load_waveform_panns(audio_path=audio_path, sample_rate=self.sample_rate, dur_seconds=self.duration_seconds)
        return waveform.squeeze(0).to(torch.float32)