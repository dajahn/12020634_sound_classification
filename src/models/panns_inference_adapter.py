from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from panns_inference import AudioTagging


class PannsInferenceEmbeddingExtractor(nn.Module):
    """
    Adapter around the panns-inference package.

    The panns-inference package exposes AudioTagging.inference(audio), which returns:
    - clipwise_output
    - embedding

    Note:
        The extraction itself is inference-only. Gradients are not propagated through panns-inference.
    """

    def __init__(
            self,
            checkpoint_path: str | Path | None = None,
            device: str = "cpu",
    ) -> None:
        super().__init__()

        self.device_name = device

        self.audio_tagger = AudioTagging(
            checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
            device=device,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract PANNs embeddings from 32 kHz mono waveforms.

        Args:
            waveform:
                Tensor with shape [batch_size, samples] or [batch_size, 1, samples].

        Returns:
            Tensor with shape [batch_size, embedding_dim].
        """
        if waveform.ndim == 3 and waveform.shape[1] == 1:
            waveform = waveform.squeeze(1)

        if waveform.ndim != 2:
            raise ValueError(
                "Expected waveform with shape [batch_size, samples] or "
                f"[batch_size, 1, samples], got {tuple(waveform.shape)}"
            )

        # panns-inference expects a NumPy array on CPU.
        audio = waveform.detach().cpu().numpy().astype(np.float32)

        # The package returns AudioSet predictions and the embedding vector.
        _, embedding = self.audio_tagger.inference(audio)

        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding)

        return embedding.to(dtype=torch.float32)