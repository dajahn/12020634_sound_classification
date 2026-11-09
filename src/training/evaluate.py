from __future__ import annotations

import argparse
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
from src.models.panns_wrapper import  build_panns_embedding_classifier


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """
    Load the project configuration file used for dataset and model settings.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return config


def get_device() -> torch.device:
    """
    Select GPU when available, otherwise use CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Use Metal Performance Shaders on macOS
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_cnn_test_dataloader(
    config_path: str | Path,
    batch_size: int,
    num_workers: int = 0,
    mode: str = "cnn",
) -> TorchDataLoader:
    """
    Build the ESC-50 test DataLoader.

    Use mode="cnn" for the CNN baseline and mode="panns" for the PANNs transfer-learning model.
    """
    dataset = ESC50Dataset(
        config_path=config_path,
        split="test",
        mode=mode,
        return_metadata=False,
    )

    return TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
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

def load_trained_cnn(
    checkpoint_path: str | Path,
    config: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """
    Load the best trained CNN baseline checkpoint.

    The model architecture is rebuilt from the config and then filled with the
    saved state dictionary.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    num_classes = int(config["dataset"]["num_classes"])
    model = build_cnn_baseline(num_classes=num_classes).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model

def load_trained_panns_embedding_classifier(
    checkpoint_path: str | Path,
    config: dict[str, Any],
    config_path: str | Path,
    device: torch.device,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Load the trained classifier that operates on saved PANNs embeddings.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    embedding_dim = int(config["panns"]["embedding_dim"])

    model = build_panns_embedding_classifier(
        config_path=config_path,
        embedding_dim=embedding_dim,
        dropout=dropout,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model

# def load_trained_panns_classifier(
#     panns_model: nn.Module,
#     checkpoint_path: str | Path,
#     config: dict[str, Any],
#     device: torch.device,
#     embedding_dim: int = 2048,
#     dropout: float = 0.3,
#     freeze_feature_extractor: bool = True,
# ) -> nn.Module:
#     """
#     Load the trained PANNs transfer-learning classifier.
#
#     The pretrained PANNs feature extractor is supplied by the caller, then wrapped
#     in the same classifier architecture used during training.
#     """
#     checkpoint_path = Path(checkpoint_path)
#
#     if not checkpoint_path.exists():
#         raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
#
#     num_classes = int(config["dataset"]["num_classes"])
#
#     model = PannsWrapper(
#         panns_model=panns_model,
#         embedding_dim=embedding_dim,
#         num_classes=num_classes,
#         dropout=dropout,
#         freeze_feature_extractor=freeze_feature_extractor,
#     ).to(device)
#
#     checkpoint = torch.load(
#         checkpoint_path,
#         map_location=device,
#         weights_only=False,
#     )
#
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.eval()
#
#     return model

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.
    """
    if y_true.size == 0:
        raise ValueError("Cannot compute accuracy for an empty target array.")

    return float((y_true == y_pred).mean())


def compute_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> float:
    """
    Compute macro F1-score without requiring additional dependencies.

    Macro F1 gives each class equal weight, which is useful because final reporting should not be dominated by any single class.
    """
    f1_scores: list[float] = []

    for class_index in range(num_classes):
        true_positive = int(
            np.logical_and(y_true == class_index, y_pred == class_index).sum()
        )
        false_positive = int(
            np.logical_and(y_true != class_index, y_pred == class_index).sum()
        )
        false_negative = int(
            np.logical_and(y_true == class_index, y_pred != class_index).sum()
        )

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative

        precision = (
            true_positive / precision_denominator
            if precision_denominator > 0
            else 0.0
        )
        recall = (
            true_positive / recall_denominator
            if recall_denominator > 0
            else 0.0
        )

        f1_denominator = precision + recall
        f1 = (
            2.0 * precision * recall / f1_denominator
            if f1_denominator > 0
            else 0.0
        )

        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Compute a confusion matrix with rows as true labels and columns as predictions.
    """
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for true_label, predicted_label in zip(y_true, y_pred, strict=True):
        confusion_matrix[int(true_label), int(predicted_label)] += 1

    return confusion_matrix


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    dataloader: TorchDataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run inference on the test set and collect predictions.

    Returns:
        True labels, predicted labels, and average inference time per example.
    """
    model.eval()

    all_true_labels: list[torch.Tensor] = []
    all_predicted_labels: list[torch.Tensor] = []

    total_inference_time_seconds = 0.0
    total_examples = 0

    for features, labels in dataloader:
        features = features.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()

        batch_start_time = time.perf_counter()

        logits = model(features)
        predictions = logits.argmax(dim=1).cpu()

        if device.type == "cuda":
            torch.cuda.synchronize()

        batch_inference_time_seconds = time.perf_counter() - batch_start_time

        batch_size = labels.size(0)
        total_inference_time_seconds += batch_inference_time_seconds
        total_examples += batch_size

        all_true_labels.append(labels.cpu())
        all_predicted_labels.append(predictions)

    y_true = torch.cat(all_true_labels).numpy()
    y_pred = torch.cat(all_predicted_labels).numpy()

    average_inference_time_seconds = total_inference_time_seconds / total_examples

    return y_true, y_pred, average_inference_time_seconds


def evaluate_cnn_baseline(
    config_path: str | Path = "configs/config.yaml",
    checkpoint_path: str | Path = "outputs/checkpoints/cnn_baseline_best.pt",
    num_workers: int = 0,
) -> dict[str, Any]:
    """
    Evaluate the trained CNN baseline on the ESC-50 test split.
    """
    config = load_config(config_path)
    training_config = config["training"]
    dataset_config = config["dataset"]

    batch_size = int(training_config["batch_size"])
    num_classes = int(dataset_config["num_classes"])

    device = get_device()

    test_loader = build_cnn_test_dataloader(
        config_path=config_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = load_trained_cnn(
        checkpoint_path=checkpoint_path,
        config=config,
        device=device,
    )

    y_true, y_pred, average_inference_time_seconds = collect_predictions(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    accuracy = compute_accuracy(y_true, y_pred)
    macro_f1 = compute_macro_f1(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=num_classes,
    )
    confusion_matrix = compute_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=num_classes,
    )

    results = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "average_inference_time_seconds": average_inference_time_seconds,
        "confusion_matrix": confusion_matrix,
    }

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test macro F1-score: {macro_f1:.4f}")
    print(
        "Average inference time per example: "
        f"{average_inference_time_seconds * 1000.0:.2f} ms"
    )
    print("Confusion matrix:")
    print(confusion_matrix)

    return results

# def evaluate_panns_classifier(
#     panns_model: nn.Module,
#     config_path: str | Path = "configs/config.yaml",
#     checkpoint_path: str | Path = "outputs/checkpoints/panns_classifier_best.pt",
#     embedding_dim: int = 2048,
#     dropout: float = 0.3,
#     freeze_feature_extractor: bool = True,
#     num_workers: int = 0,
# ) -> dict[str, Any]:
#     """
#     Evaluate the trained PANNs transfer-learning classifier on the ESC-50 test set.
#
#     This mirrors the CNN evaluation and reports:
#     - accuracy
#     - macro F1-score
#     - confusion matrix
#     - average inference time per example
#     """
#     config = load_config(config_path)
#     training_config = config["training"]
#     dataset_config = config["dataset"]
#
#     batch_size = int(training_config["batch_size"])
#     num_classes = int(dataset_config["num_classes"])
#
#     device = get_device()
#
#     test_loader = build_test_dataloader(
#         config_path=config_path,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         mode="panns",
#     )
#
#     model = load_trained_panns_classifier(
#         panns_model=panns_model,
#         checkpoint_path=checkpoint_path,
#         config=config,
#         device=device,
#         embedding_dim=embedding_dim,
#         dropout=dropout,
#         freeze_feature_extractor=freeze_feature_extractor,
#     )
#
#     y_true, y_pred, average_inference_time_seconds = collect_predictions(
#         model=model,
#         dataloader=test_loader,
#         device=device,
#     )
#
#     accuracy = compute_accuracy(y_true, y_pred)
#     macro_f1 = compute_macro_f1(
#         y_true=y_true,
#         y_pred=y_pred,
#         num_classes=num_classes,
#     )
#     confusion_matrix = compute_confusion_matrix(
#         y_true=y_true,
#         y_pred=y_pred,
#         num_classes=num_classes,
#     )
#
#     results = {
#         "accuracy": accuracy,
#         "macro_f1": macro_f1,
#         "average_inference_time_seconds": average_inference_time_seconds,
#         "confusion_matrix": confusion_matrix,
#     }
#
#     print(f"[PANNs] Test accuracy: {accuracy:.4f}")
#     print(f"[PANNs] Test macro F1-score: {macro_f1:.4f}")
#     print(
#         "[PANNs] Average inference time per example: "
#         f"{average_inference_time_seconds * 1000.0:.2f} ms"
#     )
#     print("[PANNs] Confusion matrix:")
#     print(confusion_matrix)
#
#     return results


def evaluate_panns_embedding_classifier(
    config_path: str | Path = "configs/config.yaml",
    checkpoint_path: str | Path = "outputs/checkpoints/panns_embedding_classifier_best.pt",
    num_workers: int = 0,
) -> dict[str, Any]:
    """
    Evaluate the classifier trained on precomputed PANNs embeddings.

    This uses the saved test embeddings instead of loading WAV files and running
    the PANNs backbone again.
    """
    config = load_config(config_path)
    training_config = config["training"]
    dataset_config = config["dataset"]
    panns_config = config["panns"]

    batch_size = int(training_config["batch_size"])
    num_classes = int(dataset_config["num_classes"])
    test_embeddings_path = panns_config["test_embeddings_path"]

    device = get_device()

    test_loader = build_panns_embedding_dataloader(
        embeddings_path=test_embeddings_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = load_trained_panns_embedding_classifier(
        checkpoint_path=checkpoint_path,
        config=config,
        config_path=config_path,
        device=device,
    )

    y_true, y_pred, average_inference_time_seconds = collect_predictions(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    accuracy = compute_accuracy(y_true, y_pred)
    macro_f1 = compute_macro_f1(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=num_classes,
    )
    confusion_matrix = compute_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=num_classes,
    )

    results = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "average_inference_time_seconds": average_inference_time_seconds,
        "confusion_matrix": confusion_matrix,
    }

    print(f"[PANNs Embeddings] Test accuracy: {accuracy:.4f}")
    print(f"[PANNs Embeddings] Test macro F1-score: {macro_f1:.4f}")
    print(
        "[PANNs Embeddings] Average classifier inference time per example: "
        f"{average_inference_time_seconds * 1000.0:.2f} ms"
    )
    print("[PANNs Embeddings] Confusion matrix:")
    print(confusion_matrix)

    return results

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for running evaluation as a script.
    """
    parser = argparse.ArgumentParser(description="Evaluate ESC-50 classifiers.")
    parser.add_argument(
        "--model",
        choices=("cnn", "panns-embeddings"),
        default="cnn",
        help="Which model to evaluate.",
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
        help="Path to the trained checkpoint.",
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

        evaluate_cnn_baseline(
            config_path=args.config,
            checkpoint_path=checkpoint_path,
            num_workers=args.num_workers,
        )

    elif args.model == "panns-embeddings":
        checkpoint_path = (
            args.checkpoint
            or "outputs/checkpoints/panns_embedding_classifier_best.pt"
        )

        evaluate_panns_embedding_classifier(
            config_path=args.config,
            checkpoint_path=checkpoint_path,
            num_workers=args.num_workers,
        )