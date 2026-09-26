"""
src/training/trainer.py

Consolidated training loop for RetinaGrade, replacing four near-identical
copy-pasted loops that lived in the notebook:
    1. base training        (Focal/CrossEntropy loss, frozen backbone)
    2. fine-tuning           (Focal loss, partially unfrozen backbone)
    3. CORN training         (ordinal loss, frozen backbone)
    4. CORN fine-tuning      (ordinal loss, partially unfrozen backbone)

Design notes / decisions made explicit rather than silently picked:

- The original notebook was INCONSISTENT about what metric drives
  `scheduler.step(...)`: the base run, the fine-tune run, and the CORN
  fine-tune run all step on `val_qwk` (scheduler mode='max'), but the
  first CORN run steps on `avg_val_loss` (scheduler mode='min'). Rather
  than picking one silently, `Trainer` takes `scheduler_metric` so the
  caller states which behavior they want. Default is "qwk" to match
  3 out of 4 original loops -- pass "val_loss" to reproduce the CORN
  base-training cell exactly.

- mode="corn" uses `get_predictions_corn` from evaluation.py (note: the
  module is still named with the original typo -- rename to
  evaluation.py in a later pass and update this import).

- EarlyStopping (early_stopper.py) is called as `early_stopper(score, model)`
  and always expects the metric that its own `mode` was configured for.
  All four original loops call it with `val_qwk` and mode='max', so
  Trainer does the same regardless of scheduler_metric.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Literal

import torch

from src.eval.evaluation import get_predictions, compute_metrics, compute_qwk, get_predictions_corn


@dataclass
class Trainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: object                    # torch.optim.lr_scheduler.ReduceLROnPlateau
    loss_fn: Callable                     # (outputs, labels) -> scalar tensor
    device: str
    early_stopper: Optional[object] = None
    mode: Literal["standard", "corn"] = "standard"
    num_classes: Optional[int] = None     # required when mode="corn"
    scheduler_metric: Literal["qwk", "val_loss"] = "qwk"
    verbose: bool = True

    history: dict = field(default_factory=lambda: {"train_loss": [], "val_loss": [],
                                                     "val_f1": [], "val_qwk": []})

    def __post_init__(self):
        if self.mode == "corn" and self.num_classes is None:
            raise ValueError("num_classes is required when mode='corn'")

    def _run_epoch(self, loader, train: bool) -> float:
        self.model.train() if train else self.model.eval()
        running_loss = 0.0

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)

                if train:
                    self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

                if train:
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item()

        return running_loss / len(loader)

    def _get_predictions(self, val_loader):
        if self.mode == "corn":
            return get_predictions_corn(
                self.model, val_loader, self.device, num_classes=self.num_classes
            )
        return get_predictions(self.model, val_loader, self.device)

    def fit(self, train_loader, val_loader, epochs: int):
        for epoch in range(epochs):
            avg_train_loss = self._run_epoch(train_loader, train=True)
            self.history["train_loss"].append(avg_train_loss)

            avg_val_loss = self._run_epoch(val_loader, train=False)
            self.history["val_loss"].append(avg_val_loss)

            y_pred, y_true = self._get_predictions(val_loader)
            val_metrics = compute_metrics(y_true, y_pred, average="macro")
            val_f1 = val_metrics["f1"]
            val_qwk = compute_qwk(y_true, y_pred)
            self.history["val_f1"].append(val_f1)
            self.history["val_qwk"].append(val_qwk)

            if self.verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"- train_loss: {avg_train_loss:.4f} "
                    f"- val_loss: {avg_val_loss:.4f}"
                )
                print(f"  val macro F1: {val_f1:.4f} | val QWK: {val_qwk:.4f}")

            step_value = val_qwk if self.scheduler_metric == "qwk" else avg_val_loss
            self.scheduler.step(step_value)
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.verbose:
                print(f"  current LR: {current_lr:.2e}")

            # All four original loops early-stop on val_qwk regardless of
            # what drives the scheduler -- kept identical here.
            if self.early_stopper is not None:
                self.early_stopper(val_qwk, self.model)
                if self.early_stopper.early_stop:
                    if self.verbose:
                        print(
                            f"Early stopping triggered at epoch {epoch + 1} "
                            f"(best QWK: {self.early_stopper.best_score:.4f})"
                        )
                    break

        return self.history

    def load_best_checkpoint(self):
        """Loads the early stopper's saved best weights back into self.model."""
        if self.early_stopper is None:
            raise ValueError("No early_stopper configured -- nothing to load.")
        self.model.load_state_dict(torch.load(self.early_stopper.path))
        return self.model