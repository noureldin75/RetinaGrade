import copy
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from evalution import evaluate_model

def tune_learning_rates(model, train_loader, val_loader, loss_function, device, lrs=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3], epochs=6):
    """
    Tests multiple learning rates, trains a fresh copy of the model for each, 
    tracks the best Macro F1 score, and plots the results.
    """
    results = {}

    for lr in lrs:
        print(f"\n--- Testing Learning Rate: {lr} ---")
        
        # Deepcopy the model so every LR starts from the exact same initial weights
        current_model = copy.deepcopy(model)
        current_model.to(device)
        
        optimizer = optim.Adam(current_model.classifier.parameters(), lr=lr, weight_decay=1e-2)
        
        best_f1 = -1.0
        
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
                
            # Validation check per epoch
            val_metrics = evaluate_model(
                current_model,
                val_loader,
                device,
                average="macro",
                plot=False
            )
            
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                
        results[lr] = best_f1
        print(f"LR {lr} Complete -> Best Macro F1: {best_f1:.4f}")

    # Plotting the comparison curve
    lrs_list = list(results.keys())
    f1_scores = list(results.values())
    
    plt.figure(figsize=(8, 5))
    plt.plot(lrs_list, f1_scores, marker='o', linestyle='-', color='b', linewidth=2)
    plt.xscale('log')
    plt.xlabel("Learning Rate (Log Scale)")
    plt.ylabel("Best Validation Macro F1")
    plt.title("Learning Rate Tuning Performance")
    plt.grid(True, which="both", ls="--")
    plt.show()

    return results