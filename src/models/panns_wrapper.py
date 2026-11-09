from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
import yaml


# class PannsWrapper(nn.Module):
#     """
#     Transfer-learning wrapper for a pretrained PANNs CNN14-style audio model.
#
#     This class keeps the pretrained PANNs model frozen by default and trains only a small classifier head on top of fixed-size embeddings.
#
#     Expected input shape:
#         [batch_size, num_samples]
#
#     Default ESC-50 output shape:
#         [batch_size, 50]
#
#     Notes:
#         The wrapped PANNs model is expected to return either:
#         - a tensor embedding,
#         - a dictionary containing "embedding",
#         - a dictionary containing "framewise_embedding", or
#         - a dictionary containing "clipwise_embedding".
#
#         If frame-level embeddings are returned with shape [batch, frames, dim], they are aggregated over time using mean pooling.
#     """
#
#     def __init__(
#         self,
#         panns_model: nn.Module,
#         embedding_dim: int = 2048,
#         num_classes: int = 50,
#         dropout: float = 0.3,
#         freeze_feature_extractor: bool = True,
#     ) -> None:
#         super().__init__()
#
#         self.panns_model = panns_model
#         self.embedding_dim = embedding_dim
#         self.num_classes = num_classes
#         self.freeze_feature_extractor = freeze_feature_extractor
#
#         if freeze_feature_extractor:
#             self.freeze_panns_model()
#
#         self.classifier = nn.Sequential(
#             nn.LayerNorm(embedding_dim),
#             nn.Dropout(p=dropout),
#             nn.Linear(embedding_dim, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=dropout),
#             nn.Linear(512, num_classes),
#         )
#
#         self._initialize_classifier()
#
#     def forward(self, waveform: torch.Tensor) -> torch.Tensor:
#         """
#         Compute class logits from a batch of mono 32 kHz waveforms.
#
#         Args:
#             waveform:
#                 Tensor with shape [batch_size, num_samples].
#                 If shape [batch_size, 1, num_samples] is provided, the channel dimension is removed.
#
#         Returns:
#             Class logits with shape [batch_size, num_classes].
#         """
#         embeddings = self.extract_embeddings(waveform)
#         return self.classifier(embeddings)
#
#     def extract_embeddings(self, waveform: torch.Tensor) -> torch.Tensor:
#         """
#         Extract fixed-size embeddings from the wrapped PANNs model.
#
#         Args:
#             waveform:
#                 Tensor with shape [batch_size, num_samples] or [batch_size, 1, num_samples].
#
#         Returns:
#             Tensor with shape [batch_size, embedding_dim].
#         """
#         if waveform.ndim == 3 and waveform.shape[1] == 1:
#             waveform = waveform.squeeze(1)
#
#         if waveform.ndim != 2:
#             raise ValueError("Expected waveform with shape [batch_size, num_samples] "
#                 f"or [batch_size, 1, num_samples], got {tuple(waveform.shape)}"
#             )
#
#         if self.freeze_feature_extractor:
#             with torch.no_grad():
#                 output = self.panns_model(waveform)
#         else:
#             output = self.panns_model(waveform)
#
#         embeddings = self._extract_embedding_tensor(output)
#         embeddings = self._aggregate_embeddings(embeddings)
#
#         if embeddings.ndim != 2:
#             raise ValueError("Expected aggregated embeddings with shape [batch_size, embedding_dim], "
#                 f"got {tuple(embeddings.shape)}"
#             )
#
#         if embeddings.shape[1] != self.embedding_dim:
#             raise ValueError(f"Expected embedding dimension {self.embedding_dim}, "
#                 f"got {embeddings.shape[1]}"
#             )
#
#         return embeddings.to(torch.float32)
#
#     def predict_proba(self, waveform: torch.Tensor) -> torch.Tensor:
#         """
#         Return class probabilities for inference.
#
#         Args:
#             waveform:
#                 Tensor with shape [batch_size, num_samples] or [batch_size, 1, num_samples].
#
#         Returns:
#             Softmax probabilities with shape [batch_size, num_classes].
#         """
#         logits = self.forward(waveform)
#         return torch.softmax(logits, dim=1)
#
#     def freeze_panns_model(self) -> None:
#         """
#         Freeze all parameters in the wrapped PANNs feature extractor.
#         """
#         self.panns_model.eval()
#         for parameter in self.panns_model.parameters():
#             parameter.requires_grad = False
#
#     def unfreeze_panns_model(self) -> None:
#         """
#         Unfreeze the wrapped PANNs feature extractor for optional fine-tuning.
#         """
#         self.freeze_feature_extractor = False
#         for parameter in self.panns_model.parameters():
#             parameter.requires_grad = True
#
#     @staticmethod
#     def _extract_embedding_tensor(output: Any) -> torch.Tensor:
#         """
#         Normalize common PANNs output formats to an embedding tensor.
#         """
#         if isinstance(output, torch.Tensor):
#             return output
#
#         if isinstance(output, dict):
#             for key in ("embedding", "framewise_embedding", "clipwise_embedding"):
#                 value = output.get(key)
#                 if isinstance(value, torch.Tensor):
#                     return value
#
#             available_keys = ", ".join(sorted(str(key) for key in output.keys()))
#             raise ValueError("PANNs model output dictionary does not contain a supported embedding key. Expected one of: embedding, framewise_embedding, "
#                 f"clipwise_embedding. Available keys: {available_keys}"
#             )
#
#         raise TypeError(
#             "Unsupported PANNs model output type. Expected torch.Tensor or dict, got {type(output).__name__}"
#         )
#
#     @staticmethod
#     def _aggregate_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
#         """
#         Convert frame-level embeddings to clip-level embeddings.
#
#         Supported shapes:
#             [batch_size, embedding_dim]
#             [batch_size, frames, embedding_dim]
#             [batch_size, embedding_dim, frames]
#         """
#         if embeddings.ndim == 2:
#             return embeddings
#
#         if embeddings.ndim != 3:
#             raise ValueError("Expected embeddings with shape [batch_size, embedding_dim], "
#                 f"[batch_size, frames, embedding_dim], or "
#                 f"[batch_size, embedding_dim, frames], got {tuple(embeddings.shape)}"
#             )
#
#         # PANNs framewise embeddings are usually [batch, frames, embedding_dim].
#         if embeddings.shape[-1] >= embeddings.shape[1]:
#             return embeddings.mean(dim=1)
#
#         # Also support [batch, embedding_dim, frames].
#         return embeddings.mean(dim=2)
#
#     def _initialize_classifier(self) -> None:
#         """
#         Initialize classifier head weights.
#         """
#         for module in self.classifier.modules():
#             if isinstance(module, nn.Linear):
#                 nn.init.xavier_uniform_(module.weight)
#                 nn.init.zeros_(module.bias)


def load_panns_checkpoint(
    panns_model: nn.Module,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
    strict: bool = False,
) -> nn.Module:
    """
    Load pretrained weights into a PANNs model.

    Args:
        panns_model:
            Instantiated PANNs CNN14-style model.
        checkpoint_path:
            Path to a checkpoint file.
        map_location:
            Device mapping used by torch.load.
        strict:
            Whether checkpoint keys must exactly match the model.

    Returns:
        The model with loaded weights.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"PANNs checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model")
            or checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )
    else:
        raise TypeError(
            "Unsupported checkpoint format. Expected a dictionary containing "
            "model weights."
        )

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = key.removeprefix("module.")
        cleaned_state_dict[cleaned_key] = value

    panns_model.load_state_dict(cleaned_state_dict, strict=strict)
    return panns_model


class PannsEmbeddingClassifier(nn.Module):
    """
    Classifier trained on precomputed frozen PANNs embeddings.

    This model is used after running:

        python -m src.training.precompute_panns_embeddings

    for generating/extracting the embeddings first

    Expected input shape:
        [batch_size, embedding_dim]

    Default ESC-50 output shape:
        [batch_size, 50]
    """

    def __init__(
        self,
        embedding_dim: int = 2048,
        num_classes: int = 50,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # Small classifier head trained on top of frozen PANNs embeddings.
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(p=dropout),
            nn.Linear(embedding_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        self._initialize_classifier()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute ESC-50 class logits from PANNs embeddings.

        Args:
            embeddings:
                Tensor with shape [batch_size, embedding_dim].

        Returns:
            Class logits with shape [batch_size, num_classes].
        """
        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected embeddings with shape [batch_size, embedding_dim], "
                f"got {tuple(embeddings.shape)}"
            )

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )

        return self.classifier(embeddings.to(torch.float32))

    def predict_proba(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Return class probabilities for saved PANNs embeddings.
        """
        logits = self.forward(embeddings)
        return torch.softmax(logits, dim=1)

    def _initialize_classifier(self) -> None:
        """
        Initialize classifier head weights.
        """
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)


def build_panns_embedding_classifier(
    config_path: str | Path = "configs/config.yaml",
    embedding_dim: int = 2048,
    dropout: float = 0.3,
) -> PannsEmbeddingClassifier:
    """
    Factory function for creating the classifier trained on saved PANNs embeddings.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    num_classes = int(config["dataset"]["num_classes"])

    return PannsEmbeddingClassifier(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        dropout=dropout,
    )

# def build_panns_wrapper(
#     panns_model: nn.Module,
#     config_path: str | Path = "configs/config.yaml",
#     embedding_dim: int = 2048,
#     dropout: float = 0.3,
#     freeze_feature_extractor: bool = True,
# ) -> PannsWrapper:
#     """
#     Factory function for creating the PANNs transfer-learning model.
#     """
#     config_path = Path(config_path)
#
#     if not config_path.exists():
#         raise FileNotFoundError(f"Config file not found: {config_path}")
#
#     with config_path.open("r", encoding="utf-8") as file:
#         config = yaml.safe_load(file)
#
#     if config is None:
#         raise ValueError(f"Config file is empty: {config_path}")
#
#     required_sections = {"dataset", "audio"}
#     missing_sections = required_sections.difference(config.keys())
#     if missing_sections:
#         raise ValueError(
#             f"Config file is missing required sections: {sorted(missing_sections)}"
#         )
#
#     audio_config = config["audio"]
#
#     if "sample_rate_panns" not in audio_config:
#         raise ValueError("Config file is missing audio.sample_rate_panns")
#
#     if int(audio_config["sample_rate_panns"]) != 32000:
#         raise ValueError(
#             "PANNs CNN14 expects 32 kHz audio. "
#             f"Got audio.sample_rate_panns={audio_config['sample_rate_panns']}"
#         )
#
#     num_classes = int(config["dataset"]["num_classes"])
#
#     return PannsWrapper(
#         panns_model=panns_model,
#         embedding_dim=embedding_dim,
#         num_classes=num_classes,
#         dropout=dropout,
#         freeze_feature_extractor=freeze_feature_extractor,
#     )

