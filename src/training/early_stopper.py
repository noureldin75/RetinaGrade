import torch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, path='best_model.pt', mode='min'):

        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score, model):
        if self.best_score is None:
            improved = True
        elif self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:  # mode == 'max'
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            print(f"  -> Best model saved ({self.mode}: {score:.4f})")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True