from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """
    Convolutional block used by the compact CNN baseline.

    Each block applies:
    - Conv2d
    - BatchNorm2d
    - ReLU
    - Conv2d
    - BatchNorm2d
    - ReLU
    - MaxPool2d
    - Dropout2d

    Input shape:
        [batch_size, in_channels, n_mels, time_frames]

    Output shape:
        [batch_size, out_channels, reduced_n_mels, reduced_time_frames]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNBaseline(nn.Module):
    """
    Compact CNN baseline for classification from normalized log-mel spectrograms.

    The model is intended for the CNN pipeline
    audio -> normalized log-mel spectrogram -> compact CNN -> 50-class logits.

    Expected input shape:
        [batch_size, 1, n_mels, time_frames]

    Default ESC-50 output shape:
        [batch_size, 50]
    """

    def __init__(
        self,
        num_classes: int = 50,
        input_channels: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(input_channels, 32, dropout=0.1),
            ConvBlock(32, 64, dropout=0.15),
            ConvBlock(64, 128, dropout=0.2),
            ConvBlock(128, 256, dropout=0.25),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute class logits for a batch of log-mel spectrograms.

        Args:
            x:
                Tensor with shape [batch_size, 1, n_mels, time_frames].

        Returns:
            Class logits with shape [batch_size, num_classes].
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected input with shape [batch_size, channels, n_mels, time_frames], "
                f"got {tuple(x.shape)}"
            )

        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return class probabilities for inference.

        Args:
            x:
                Tensor with shape [batch_size, 1, n_mels, time_frames].

        Returns:
            Softmax probabilities with shape [batch_size, num_classes].
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def _initialize_weights(self) -> None:
        """
        Initialize convolutional and linear layers with common CNN defaults.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)


def build_cnn_baseline(
    num_classes: int = 50,
    input_channels: int = 1,
    dropout: float = 0.3,
) -> CNNBaseline:
    """
    Factory function for creating the compact CNN baseline.
    """
    return CNNBaseline(
        num_classes=num_classes,
        input_channels=input_channels,
        dropout=dropout,
    )
