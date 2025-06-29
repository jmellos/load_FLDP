import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)
from load_data import load_data

# 1) Define your PyTorch forecasting model
class FeedForwardForecast(nn.Module):
    def __init__(self, input_size: int, units: int = 32, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(units, units // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(units // 2, 1),
        )

    def forward(self, x):
        # return shape (batch,) rather than (batch,1) to match target shape
        return self.net(x).squeeze(-1)

# 2) Training + early stopping loop
def train_with_early_stopping(
    model, optimizer, criterion,
    train_loader, val_loader,
    device,
    max_epochs=50, patience=10
):
    model.to(device)
    best_val_loss = float("inf")
    best_state = None
    trigger = 0

    for epoch in range(1, max_epochs + 1):
        # — Training —
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device).squeeze(-1)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

        # — Validation —
        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv, yv = Xv.to(device), yv.to(device).squeeze(-1)
                pv = model(Xv)
                b = yv.size(0)
                val_loss += criterion(pv, yv).item() * b
                n += b
        val_loss /= n

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience:
                print(f"Stopping at epoch {epoch} (no improvement in {patience} rounds)")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

# 3) Main loop over partitions (clients)
if __name__ == "__main__":
    num_partitions = 5
    batch_size     = 32
    dropout = 0.0
    lookback=12
    units=32
    lr=0.001
    max_epochs=100
    patience=10

    os.makedirs("dl_predictions", exist_ok=True)
    metrics = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for partition_id in range(num_partitions):
        print(f"\n=== Partition {partition_id} ===")
        base_dir = f"dl_predictions/client_{partition_id}"
        os.makedirs(base_dir, exist_ok=True)

        # Load data
        train_loader, test_loader, scaler_X, scaler_y = load_data(
            partition_id   = partition_id,
            num_partitions = num_partitions,
            batch_size     = batch_size,
            shuffle        = False,
            lookback=lookback
        )

        # Carve out a validation split (20% of training)
        full_train_ds = train_loader.dataset
        n_total = len(full_train_ds)
        n_val = int(0.2 * n_total)
        n_train = n_total - n_val

        train_ds = torch.utils.data.Subset(full_train_ds, range(n_train))
        val_ds   = torch.utils.data.Subset(full_train_ds, range(n_train, n_total))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        # Build model, optimizer, loss
        model     = FeedForwardForecast(input_size=lookback*4, units=32, dropout=dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Train with early stopping
        train_with_early_stopping(
            model, optimizer, criterion,
            train_loader, val_loader,
            device,
            max_epochs=max_epochs,
            patience=patience,
        )

        # Evaluate on test set
        model.eval()
        preds_scaled = []
        trues_scaled = []
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb, yb = Xb.to(device), yb.to(device).squeeze(-1)
                out = model(Xb)
                preds_scaled.append(out.cpu().numpy())
                trues_scaled.append(yb.cpu().numpy())

        preds_scaled = np.concatenate(preds_scaled, axis=0).reshape(-1, 1)
        trues_scaled = np.concatenate(trues_scaled, axis=0).reshape(-1, 1)
        y_pred = scaler_y.inverse_transform(preds_scaled).flatten()
        y_true = scaler_y.inverse_transform(trues_scaled).flatten()

        # Compute metrics
        mae  = mean_absolute_error(y_true, y_pred)
        mse  = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        metrics.append({
            "Client": f"C{partition_id+1}",
            "MAE": mae, "MSE": mse, "RMSE": rmse,
            "MAPE": mape, "R2": r2
        })
        print(f"  MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

        # Plot full & zoom
        for tag, yp, yt in [("full", y_pred, y_true),
                            ("zoom", y_pred[:100], y_true[:100])]:
            plt.figure(figsize=(12,5))
            plt.plot(yt, label="True", linewidth=2)
            plt.plot(yp, '--', label="Predicted")
            plt.title(f"Client {partition_id} – DL Smoothed ({tag})")
            plt.xlabel("Time step"); plt.ylabel("Energy (kWh)")
            plt.legend(); plt.grid(True); plt.tight_layout()
            plt.savefig(f"{base_dir}/{tag}_prediction.png")
            plt.close()

    # Save metrics CSV
    pd.DataFrame(metrics).to_csv("metrics_per_client_DL_smoothed.csv", index=False)
    print("\n✅ Done! Metrics + plots saved.")
