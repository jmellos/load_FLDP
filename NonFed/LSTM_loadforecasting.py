import os
import copy
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)

from load_data import load_data  # seu loader com lookback

# parâmetros\ nnum_clients = 5
look_back   = 12
n_features  = 4
batch_size  = 32
num_clients = 5
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pasta de saída
base_folder = "lstm_predictions"
os.makedirs(base_folder, exist_ok=True)

metrics = []

class LSTMForecast(nn.Module):
    def __init__(self, n_features, hidden_size=64, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = torch.relu(self.fc2(out))
        out = self.dropout(out)
        return self.fc3(out)

for client_id in range(num_clients):
    client_folder = os.path.join(base_folder, f"client_{client_id}")
    os.makedirs(client_folder, exist_ok=True)

    # 1) carregar dados (80% train+val, 20% test)
    train_loader, test_loader, scaler_X, scaler_y = load_data(
        partition_id=client_id,
        num_partitions=num_clients,
        batch_size=batch_size,
        shuffle=False,
        lookback=look_back,
    )

    # 2) dividir train+val em 64% train e 16% val
    full_ds = train_loader.dataset
    n_total = len(full_ds)
    n_val   = int(0.2 * n_total)
    n_train = n_total - n_val

    train_ds = Subset(full_ds, range(n_train))
    val_ds   = Subset(full_ds, range(n_train, n_total))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # 3) instanciar modelo, otimizador e loss
    model     = LSTMForecast(n_features=n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 4) treino com early stopping (patience=10, max_epochs=100)
    best_val   = float("inf")
    best_state = None
    trigger    = 0
    patience   = 10

    for epoch in range(1, 101):
        model.train()
        for Xb, yb in train_dl:
            Xb = Xb.view(-1, look_back, n_features).to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in val_dl:
                Xb = Xb.view(-1, look_back, n_features).to(device)
                yb = yb.to(device)
                val_losses.append(criterion(model(Xb), yb).item())
        val_loss = np.mean(val_losses)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = copy.deepcopy(model.state_dict())
            trigger    = 0
        else:
            trigger += 1
            if trigger >= patience:
                break

    # 5) restaurar melhores pesos
    if best_state is not None:
        model.load_state_dict(best_state)

    # 6) prever no test set
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.view(-1, look_back, n_features).to(device)
            preds.append(model(Xb).cpu().numpy())
            trues.append(yb.cpu().numpy())

    y_pred_scaled = np.vstack(preds)
    y_true_scaled = np.vstack(trues)

    # 7) inversão da padronização
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_true_scaled)

    # 8) calcular métricas
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    metrics.append({
        "Client": f"C{client_id+1}",
        "MAE": mae, "MSE": mse, "RMSE": rmse,
        "MAPE": mape, "R2": r2
    })

    # 9) salvar plots (full & zoom)
    for tag, limit in [("full", None), ("zoom", 100)]:
        plt.figure(figsize=(12,5))
        end = None if limit is None else limit
        plt.plot(y_true[:end], label="True", linewidth=2)
        plt.plot(y_pred[:end], linestyle="--", label="Predicted")
        plt.title(f"Client {client_id} – LSTM ({tag})")
        plt.xlabel("Time step")
        plt.ylabel("Energy (kWh)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(client_folder, f"{tag}_prediction.png"))
        plt.close()

# 10) salvar CSV de métricas
pd.DataFrame(metrics).to_csv(
    os.path.join(base_folder, "metrics_per_client_LSTM.csv"),
    index=False
)

print("✅ LSTM done! Metrics + plots saved.")
