import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report,
)


CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def get_predictions(model, data_loader, device):
    """
    Runs the model over a DataLoader in eval mode and collects
    predictions + true labels as numpy arrays.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return all_preds, all_labels


def compute_metrics(y_true, y_pred, average="macro"):
    """
    Computes accuracy, F1, recall, precision.
    average='macro' treats every class equally regardless of size —
    important here since Severe/Mild are small and we care about them
    as much as No DR. Use 'weighted' instead if you want overall
    performance weighted by class frequency.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
    }
    return metrics


def print_metrics(y_true, y_pred, average="macro"):
    metrics = compute_metrics(y_true, y_pred, average=average)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1 ({average}):     {metrics['f1']:.4f}")
    print(f"Recall ({average}): {metrics['recall']:.4f}")
    print(f"Precision ({average}): {metrics['precision']:.4f}")
    print("\nPer-class report:")
    print(classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, zero_division=0
    ))
    return metrics


def plot_confusion_matrix(y_true, y_pred, normalize=False, title="Confusion Matrix"):
    """
    normalize=True shows row-wise percentages (per true class) instead
    of raw counts — usually clearer for imbalanced classes like yours,
    since raw counts make rare classes look "empty" next to No DR.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        fmt = "d"

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_training_curve(losses, title="Training Loss"):
    """
    losses: list of average loss per epoch, collected during training.
    """
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def evaluate_model(model, data_loader, device, average="macro", plot=True):
    """
    Convenience wrapper: runs predictions, prints metrics, and plots
    the confusion matrix in one call.
    """
    y_pred, y_true = get_predictions(model, data_loader, device)
    metrics = print_metrics(y_true, y_pred, average=average)
    if plot:
        plot_confusion_matrix(y_true, y_pred, normalize=True)
    return metrics