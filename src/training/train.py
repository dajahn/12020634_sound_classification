from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader

from src.data_set import ESC50Dataset, PannsEmbeddingDataset
from src.models.cnn_baseline import build_cnn_baseline
from src.models.panns_wrapper import  build_panns_embedding_classifier, PannsEmbeddingClassifier


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """
    Load the project configuration file.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return config


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # These settings prefer reproducibility to maximum training speed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Select GPU when available, otherwise fall back to CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Use Metal Performance Shaders on macOS
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_cnn_dataloader(
    split: str,
    config_path: str | Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    mode: str = "cnn",
) -> TorchDataLoader:
    """
    Build a PyTorch DataLoader using the existing ESC50Dataset class.

    The dataset object reads the config file, applies the requested fold split,
    and uses either:
    - mode="cnn" for log-mel spectrogram training
    - mode="panns" for 32 kHz waveform training with PANNs embeddings
    """
    dataset = ESC50Dataset(
        config_path=config_path,
        split=split,
        mode=mode,
        return_metadata=False,
    )

    return TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

def build_panns_embedding_dataloader(
    embeddings_path: str | Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> TorchDataLoader:
    """
    Build a DataLoader for saved PANNs embeddings.

    This is much faster than extracting PANNs embeddings again during every epoch.
    """
    dataset = PannsEmbeddingDataset(embeddings_path)

    return TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_train_epoch(
    model: nn.Module,
    dataloader: TorchDataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run one training epoch.

    Returns:
        Average loss and accuracy for the epoch.
    """
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    for features, labels in dataloader:
        # Move the log-mel spectrograms and labels to the selected device.
        features = features.to(device)
        labels = labels.to(device)

        # Reset gradients from the previous optimization step.
        optimizer.zero_grad(set_to_none=True)

        # Forward pass: CNN returns unnormalized class logits.
        logits = model(features)

        # CrossEntropyLoss expects raw logits and integer class labels.
        loss = criterion(logits, labels)

        # Backpropagate and update model weights.
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        predictions = logits.argmax(dim=1)

        total_loss += loss.item() * batch_size
        correct_predictions += (predictions == labels).sum().item()
        total_examples += batch_size

    average_loss = total_loss / total_examples
    accuracy = correct_predictions / total_examples

    return average_loss, accuracy


@torch.no_grad()
def run_validation_epoch(
    model: nn.Module,
    dataloader: TorchDataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run one validation epoch without gradient tracking.

    Returns:
        Average validation loss and validation accuracy.
    """
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        predictions = logits.argmax(dim=1)

        total_loss += loss.item() * batch_size
        correct_predictions += (predictions == labels).sum().item()
        total_examples += batch_size

    average_loss = total_loss / total_examples
    accuracy = correct_predictions / total_examples

    return average_loss, accuracy


def save_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_validation_accuracy: float,
    config: dict[str, Any],
) -> None:
    """
    Save the best model checkpoint.

    The checkpoint includes the model state, optimizer state, epoch, best validation score, and config so
    that evaluation can reproduce the experiment setup.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_validation_accuracy": best_validation_accuracy,
            "config": config,
        },
        checkpoint_path,
    )


def train_cnn_baseline(
    config_path: str | Path = "configs/config.yaml",
    checkpoint_path: str | Path = "outputs/checkpoints/cnn_baseline_best.pt",
    num_workers: int = 0,
) -> dict[str, Any]:
    """
    Train the compact CNN baseline

    The process is:
    1. Load config.
    2. Build train and validation datasets
    3. Build the CNN baseline model.
    4. Train for the configured number of epochs.
    5. Save the checkpoint with the best validation accuracy.
    """
    config = load_config(config_path)
    training_config = config["training"]
    dataset_config = config["dataset"]

    seed = int(training_config["seed"])
    batch_size = int(training_config["batch_size"])
    learning_rate = float(training_config["learning_rate"])
    epochs = int(training_config["epochs"])
    num_classes = int(dataset_config["num_classes"])

    set_seed(seed)
    device = get_device()

    print(f"Using device: {device}")

    train_loader = build_cnn_dataloader(
        split="train",
        config_path=config_path,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    validation_loader = build_cnn_dataloader(
        split="validation",
        config_path=config_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = build_cnn_baseline(num_classes=num_classes).to(device)

    # CrossEntropyLoss is suitable for single-label multi-class classification.
    criterion = nn.CrossEntropyLoss()

    # Adam is a stable default optimizer for the compact CNN baseline.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    best_validation_accuracy = 0.0
    history: list[dict[str, float | int]] = []

    training_start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.perf_counter()

        train_loss, train_accuracy = run_train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        validation_loss, validation_accuracy = run_validation_epoch(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
        )

        epoch_duration_seconds = time.perf_counter() - epoch_start_time

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "epoch_duration_seconds": epoch_duration_seconds,
        }
        history.append(epoch_metrics)

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} | "
            f"val_loss={validation_loss:.4f} | "
            f"val_acc={validation_accuracy:.4f} | "
            f"time={epoch_duration_seconds:.2f}s"
        )

        # Keep the checkpoint that performs best on the validation fold.
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_validation_accuracy=best_validation_accuracy,
                config=config,
            )
            print(f"Saved new best checkpoint to: {checkpoint_path}")

    total_training_time_seconds = time.perf_counter() - training_start_time

    result = {
        "checkpoint_path": str(checkpoint_path),
        "best_validation_accuracy": best_validation_accuracy,
        "total_training_time_seconds": total_training_time_seconds,
        "history": history,
    }

    print(
        "Training finished | "
        f"best_val_acc={best_validation_accuracy:.4f} | "
        f"total_time={total_training_time_seconds:.2f}s"
    )

    return result

def train_panns_embedding_classifier(
    config_path: str | Path = "configs/config.yaml",
    checkpoint_path: str | Path = "outputs/checkpoints/panns_embedding_classifier_best.pt",
    dropout: float = 0.3,
    num_workers: int = 0,
) -> dict[str, Any]:
    """
    Train an ESC-50 classifier on precomputed frozen PANNs embeddings.

    The process is:
    1. Load train and validation embeddings from data/preprocessed/panns_embeddings.
    2. Build a small classifier head.
    3. Train only this classifier head.
    4. Save the best checkpoint according to validation accuracy.
    """
    config = load_config(config_path)
    training_config = config["training"]
    panns_config = config["panns"]

    seed = int(training_config["seed"])
    batch_size = int(training_config["batch_size"])
    learning_rate = float(training_config["learning_rate"])
    epochs = int(training_config["epochs"])
    embedding_dim = int(panns_config["embedding_dim"])

    train_embeddings_path = panns_config["train_embeddings_path"]
    validation_embeddings_path = panns_config["validation_embeddings_path"]

    set_seed(seed)
    device = get_device()

    train_loader = build_panns_embedding_dataloader(
        embeddings_path=train_embeddings_path,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    validation_loader = build_panns_embedding_dataloader(
        embeddings_path=validation_embeddings_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = build_panns_embedding_classifier(
        config_path=config_path,
        embedding_dim=embedding_dim,
        dropout=dropout,
    ).to(device)

    # ESC-50 is a single-label multi-class task.
    criterion = nn.CrossEntropyLoss()

    # Only the small embedding classifier is trained here.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    best_validation_accuracy = 0.0
    history: list[dict[str, float | int]] = []

    training_start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.perf_counter()

        train_loss, train_accuracy = run_train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        validation_loss, validation_accuracy = run_validation_epoch(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
        )

        epoch_duration_seconds = time.perf_counter() - epoch_start_time

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "epoch_duration_seconds": epoch_duration_seconds,
        }
        history.append(epoch_metrics)

        print(
            f"[PANNs Embeddings] Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} | "
            f"val_loss={validation_loss:.4f} | "
            f"val_acc={validation_accuracy:.4f} | "
            f"time={epoch_duration_seconds:.2f}s"
        )

        # Save the classifier with the best validation accuracy.
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_validation_accuracy=best_validation_accuracy,
                config=config,
            )
            print(f"Saved new best PANNs embedding checkpoint to: {checkpoint_path}")

    total_training_time_seconds = time.perf_counter() - training_start_time

    result = {
        "checkpoint_path": str(checkpoint_path),
        "best_validation_accuracy": best_validation_accuracy,
        "total_training_time_seconds": total_training_time_seconds,
        "history": history,
    }

    print(
        "[PANNs Embeddings] Training finished | "
        f"best_val_acc={best_validation_accuracy:.4f} | "
        f"total_time={total_training_time_seconds:.2f}s"
    )

    return result

# def train_panns_classifier(
#     panns_model: nn.Module,
#     config_path: str | Path = "configs/config.yaml",
#     checkpoint_path: str | Path = "outputs/checkpoints/panns_classifier_best.pt",
#     embedding_dim: int = 2048,
#     dropout: float = 0.3,
#     freeze_feature_extractor: bool = True,
#     num_workers: int = 0,
# ) -> dict[str, Any]:
#     """
#     Train the PANNs transfer-learning classifier described in the README.
#
#     The process is:
#     1. Load configuration.
#     2. Build train and validation datasets in mode="panns".
#     3. Wrap a pretrained PANNs CNN14-style model with PannsWrapper.
#     4. Freeze the PANNs feature extractor by default.
#     5. Train only the classifier head on ESC-50 labels.
#     6. Save the best checkpoint according to validation accuracy.
#
#     Args:
#         panns_model:
#             A pretrained PANNs CNN14-style PyTorch model. The project wrapper expects
#             this model to return an embedding tensor or a dictionary containing an
#             embedding-like tensor.
#         config_path:
#             Path to config.yaml.
#         checkpoint_path:
#             Output path for the best validation checkpoint.
#         embedding_dim:
#             Size of the PANNs embedding vector. CNN14 commonly uses 2048.
#         dropout:
#             Dropout probability used in the classifier head.
#         freeze_feature_extractor:
#             If True, only the classifier head is trained.
#         num_workers:
#             Number of PyTorch DataLoader worker processes.
#
#     Returns:
#         Dictionary containing training history and the best validation accuracy.
#     """
#     config = load_config(config_path)
#     training_config = config["training"]
#     dataset_config = config["dataset"]
#
#     seed = int(training_config["seed"])
#     batch_size = int(training_config["batch_size"])
#     learning_rate = float(training_config["learning_rate"])
#     epochs = int(training_config["epochs"])
#     num_classes = int(dataset_config["num_classes"])
#
#     set_seed(seed)
#     device = get_device()
#
#     train_loader = build_dataloader(
#         split="train",
#         config_path=config_path,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         mode="panns",
#     )
#     validation_loader = build_dataloader(
#         split="validation",
#         config_path=config_path,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         mode="panns",
#     )
#
#     model = PannsWrapper(
#         panns_model=panns_model,
#         embedding_dim=embedding_dim,
#         num_classes=num_classes,
#         dropout=dropout,
#         freeze_feature_extractor=freeze_feature_extractor,
#     ).to(device)
#
#     # CrossEntropyLoss is used because ESC-50 is a single-label 50-class task.
#     criterion = nn.CrossEntropyLoss()
#
#     # If the feature extractor is frozen, this optimizer effectively trains only
#     # the classifier head because frozen parameters have requires_grad=False.
#     optimizer = torch.optim.Adam(
#         filter(lambda parameter: parameter.requires_grad, model.parameters()),
#         lr=learning_rate,
#     )
#
#     best_validation_accuracy = 0.0
#     history: list[dict[str, float | int]] = []
#
#     training_start_time = time.perf_counter()
#
#     for epoch in range(1, epochs + 1):
#         epoch_start_time = time.perf_counter()
#
#         train_loss, train_accuracy = run_train_epoch(
#             model=model,
#             dataloader=train_loader,
#             criterion=criterion,
#             optimizer=optimizer,
#             device=device,
#         )
#         validation_loss, validation_accuracy = run_validation_epoch(
#             model=model,
#             dataloader=validation_loader,
#             criterion=criterion,
#             device=device,
#         )
#
#         epoch_duration_seconds = time.perf_counter() - epoch_start_time
#
#         epoch_metrics = {
#             "epoch": epoch,
#             "train_loss": train_loss,
#             "train_accuracy": train_accuracy,
#             "validation_loss": validation_loss,
#             "validation_accuracy": validation_accuracy,
#             "epoch_duration_seconds": epoch_duration_seconds,
#         }
#         history.append(epoch_metrics)
#
#         print(
#             f"[PANNs] Epoch {epoch:03d}/{epochs:03d} | "
#             f"train_loss={train_loss:.4f} | "
#             f"train_acc={train_accuracy:.4f} | "
#             f"val_loss={validation_loss:.4f} | "
#             f"val_acc={validation_accuracy:.4f} | "
#             f"time={epoch_duration_seconds:.2f}s"
#         )
#
#         # Store the best transfer-learning classifier based on validation accuracy.
#         if validation_accuracy > best_validation_accuracy:
#             best_validation_accuracy = validation_accuracy
#             save_checkpoint(
#                 checkpoint_path=checkpoint_path,
#                 model=model,
#                 optimizer=optimizer,
#                 epoch=epoch,
#                 best_validation_accuracy=best_validation_accuracy,
#                 config=config,
#             )
#             print(f"Saved new best PANNs checkpoint to: {checkpoint_path}")
#
#     total_training_time_seconds = time.perf_counter() - training_start_time
#
#     result = {
#         "checkpoint_path": str(checkpoint_path),
#         "best_validation_accuracy": best_validation_accuracy,
#         "total_training_time_seconds": total_training_time_seconds,
#         "history": history,
#     }
#
#     print(
#         "[PANNs] Training finished | "
#         f"best_val_acc={best_validation_accuracy:.4f} | "
#         f"total_time={total_training_time_seconds:.2f}s"
#     )
#
#     return result


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for running training as a script.
    """
    parser = argparse.ArgumentParser(description="Train ESC-50 classifiers.")
    parser.add_argument(
        "--model",
        choices=("cnn", "panns-embeddings"),
        default="cnn",
        help="Which model to train.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Where to save the best validation checkpoint.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes for PyTorch DataLoader.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.model == "cnn":
        checkpoint_path = args.checkpoint or "outputs/checkpoints/cnn_baseline_best.pt"

        train_cnn_baseline(
            config_path=args.config,
            checkpoint_path=checkpoint_path,
            num_workers=args.num_workers,
        )

    elif args.model == "panns-embeddings":
        checkpoint_path = (
            args.checkpoint
            or "outputs/checkpoints/panns_embedding_classifier_best.pt"
        )

        train_panns_embedding_classifier(
            config_path=args.config,
            checkpoint_path=checkpoint_path,
            num_workers=args.num_workers,
        )