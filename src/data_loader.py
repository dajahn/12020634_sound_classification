from pathlib import Path
import torch
import torchaudio
import soundfile as sf

class AudioLoadingError(RuntimeError):
    """Raised when an audio file cannot be loaded or preprocessed."""


class DataLoader:

    @staticmethod
    def _load_audio_file(audio_path: Path) -> tuple[torch.Tensor, int]:
        """
        Load an audio file without relying on TorchCodec.

        soundfile handles ESC-50 WAV files directly and avoids TorchCodec/FFmpeg
        DLL issues on Windows. torchaudio is kept as a fallback.
        """
        try:
            audio_data, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
            waveform = torch.from_numpy(audio_data.T.copy())
            return waveform, sample_rate
        except Exception as soundfile_ex:
            try:
                return torchaudio.load(str(audio_path))
            except Exception as torchaudio_ex:
                raise AudioLoadingError(
                    f"Error while loading audio file '{audio_path}'. "
                    f"soundfile error: {soundfile_ex}. "
                    f"torchaudio error: {torchaudio_ex}"
                ) from torchaudio_ex

    @staticmethod
    def load_waveform(
            audio_path: str|Path,
            target_sample_rate: int,
            duration_seconds: float = 5.0,
            mono: bool = True
    ) -> torch.Tensor:
        """
            Load a WAV file and return a waveform tensor with a fixed duration.
            Steps:
            1. Load WAV file.
            2. Convert stereo/multi-channel audio to mono if requested.
            3. Resample to the target sample rate if needed.
            4. Pad or trim to the target duration.

            Args:
                audio_path:
                    Path to the WAV file.
                target_sample_rate:
                    Required output sample rate.
                    Use 44100 for the CNN baseline and 32000 for PANNs.
                duration_seconds:
                    Required output duration in seconds.
                mono:
                    Whether to convert multi-channel audio to mono.

            Returns:
                A waveform tensor with shape [1, num_samples] if mono=True.

            Raises:
                AudioLoadingError:
                    If the file does not exist, cannot be decoded, or produces invalid audio.
            """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise AudioLoadingError(f"Audio file does not exist: {audio_path}")
        if not audio_path.is_file():
            raise AudioLoadingError(f"Audio file is not a file: {audio_path}")

        waveform, original_sample_rate = DataLoader._load_audio_file(audio_path)

        if waveform.numel() == 0:
            raise AudioLoadingError(f"Audio file is empty: {audio_path}")
        if waveform.ndim != 2:
            raise AudioLoadingError(f"Expected waveform with shape [channels, samples], got {tuple(waveform.shape)} "
                                    f"for file: {audio_path}")

        if torch.isnan(waveform).any() or torch.isinf(waveform).any():
            raise AudioLoadingError(f"Audio file contains NaN or some infinite values: {waveform}")

        if mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if original_sample_rate != target_sample_rate:
            try:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=original_sample_rate,
                    new_freq=target_sample_rate
                )
                waveform = resampler(waveform)
            except Exception as ex:
                raise AudioLoadingError(f"Error resampling audio file {audio_path} from {original_sample_rate} Hz to {target_sample_rate} Hz: {ex}") from ex

        target_num_samples = int(target_sample_rate * duration_seconds)
        waveform = DataLoader.pad_or_trim_waveform(waveform, target_num_samples)

        if waveform.shape[-1] != target_num_samples:
            raise AudioLoadingError(f"Audio preprocessing failed for {audio_path} Expected {target_num_samples}, got {waveform.shape[-1]}")
        return waveform

    @staticmethod
    def pad_or_trim_waveform(waveform: torch.Tensor, target_num_samples: int) -> torch.Tensor:
        """
            Pad or trim a waveform to a fixed number of samples.
            Args:
                waveform:
                    Tensor with shape [channels, samples].
                target_num_samples:
                    Required number of samples.

            Returns:
                Tensor with shape [channels, target_num_samples].
            """
        curr_num_samples = waveform.shape[-1]
        if curr_num_samples > target_num_samples:
            return waveform[..., :target_num_samples]
        if curr_num_samples < target_num_samples:
            padding_amount = target_num_samples - curr_num_samples
            return torch.nn.functional.pad(waveform, (0, padding_amount))
        return waveform

    @staticmethod
    def load_waveform_cnn(
            audio_path: str | Path,
            sample_rate: int = 44100,
            dur_seconds: float = 5.0
    ) -> torch.Tensor:
        """
           Load audio for the CNN log-mel spectrogram pipeline.

           ESC-50 audio is released at 44.1 kHz, so 44100 Hz is a reasonable default.
           """
        return DataLoader.load_waveform(audio_path= audio_path, target_sample_rate=sample_rate, duration_seconds=dur_seconds, mono=True)

    @staticmethod
    def load_waveform_panns(
            audio_path: str | Path,
            sample_rate: int = 32000,
            dur_seconds: float = 5.0
    ) -> torch.Tensor:
        """
           Load audio for the PANNs CNN14 pipeline.

           PANNs CNN14 expects 32 kHz mono audio.
           """
        return DataLoader.load_waveform(audio_path=audio_path, target_sample_rate=sample_rate, duration_seconds=dur_seconds, mono=True)