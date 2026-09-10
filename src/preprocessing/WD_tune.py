import copy
import torch
import torch.optim as optim

from evalution import evaluate_model


def tune_weight_decay(
    model,
    train_loader,
    val_loader,
    loss_function,
    device,
    lr,
    weight_decays,
    epochs=6,
    trainable_module_name="classifier",
):
    """
    Sweeps weight_decay values while keeping lr fixed, mirroring
    tune_learning_rates. Only the parameters of `trainable_module_name`
    are optimized (matches your current setup: backbone frozen,
    classifier head trainable).

    Trains a fresh deepcopy of `model` for every trial, so the original
    `model` object passed in is never mutated and every weight_decay
    value starts from the exact same initial weights.

    Returns
    -------
    dict: {weight_decay_value: {"best_val_f1": float, "history": [...]}}
    """
    results = {}

    for wd in weight_decays:
        print(f"\n=== weight_decay = {wd} (lr fixed at {lr}) ===")

        # Fresh copy of the whole model for this trial — the original
        # `model` passed in is never touched.
        current_model = copy.deepcopy(model)
        current_model.to(device)
        trainable_module = getattr(current_model, trainable_module_name)

        optimizer = optim.Adam(
            trainable_module.parameters(),
            lr=lr,
            weight_decay=wd,
        )

        history = []
        best_val_f1 = -1.0

        for epoch in range(epochs):
            current_model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = current_model(images)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)

            val_metrics = evaluate_model(
                current_model, val_loader, device, average="macro", plot=False
            )

            print(
                f"  Epoch {epoch + 1}/{epochs} "
                f"- train_loss: {avg_train_loss:.4f} "
                f"- val_f1: {val_metrics['f1']:.4f}"
            )

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_f1": val_metrics["f1"],
                }
            )

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]

        results[wd] = {"best_val_f1": best_val_f1, "history": history}
        print(f"  -> best val macro F1 for wd={wd}: {best_val_f1:.4f}")

    best_wd = max(results, key=lambda k: results[k]["best_val_f1"])
    print("\n" + "=" * 50)
    print(f"BEST weight_decay overall: {best_wd}")
    print(f"BEST val macro F1: {results[best_wd]['best_val_f1']:.4f}")
    print("=" * 50)

    return results