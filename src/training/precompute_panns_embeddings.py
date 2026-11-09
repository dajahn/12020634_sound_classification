from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader as TorchDataLoader

from src.data_set import ESC50Dataset
from src.models.panns_inference_adapter import PannsInferenceEmbeddingExtractor


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load the project configuration.
    """
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return config


@torch.no_grad()
def extract_split_embeddings(
    split: str,
    config_path: str | Path,
    output_path: str | Path,
    batch_size: int,
    checkpoint_path: str | Path | None,
    device: str,
    num_workers: int = 0,
) -> None:
    """
    Extract and save PANNs embeddings for one ESC-50 split.
    """
    dataset = ESC50Dataset(
        config_path=config_path,
        split=split,
        mode="panns",
        return_metadata=False,
    )

    dataloader = TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    extractor = PannsInferenceEmbeddingExtractor(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    all_embeddings: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for waveforms, labels in dataloader:
        embeddings = extractor(waveforms)

        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels.cpu())

        print(f"Processed {split}: {sum(x.size(0) for x in all_labels)} samples")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "embeddings": torch.cat(all_embeddings, dim=0),
            "labels": torch.cat(all_labels, dim=0),
        },
        output_path,
    )

    print(f"Saved {split} embeddings to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute PANNs embeddings.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    if torch.backends.mps.is_available():
        device = "mps"  # Use Metal Performance Shaders on macOS
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = int(config["training"]["batch_size"])
    checkpoint_path = config.get("panns", {}).get("checkpoint_path")

    if checkpoint_path in (None, "null", ""):
        checkpoint_path = None

    for split in ("train", "validation", "test"):
        extract_split_embeddings(
            split=split,
            config_path=args.config,
            output_path=f"data/preprocessed/panns_embeddings/{split}.pt",
            batch_size=batch_size,
            checkpoint_path=checkpoint_path,
            device=device,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()