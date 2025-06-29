import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

def load_data(
    partition_id: int,
    lookback: int = 12,
    test_size: float = 0.2,
):
    """
    Carrega dados do cliente, gera janelas de look-back,
    faz split temporal e padroniza.
    Retorna X_train, X_test, y_train, y_test, scaler_X, scaler_y.
    """
    client_files = {
        0: "Residential_4.csv",
        1: "Residential_8.csv",
        2: "Residential_9.csv",
        3: "Residential_10.csv",
        4: "Residential_13.csv",
    }
    # 1) leitura e limpeza
    df = pd.read_csv(client_files[partition_id])
    df["energy_kWh"] = df["energy_kWh"].interpolate(method="linear")
    df["energy_kWh"] = df["energy_kWh"].rolling(window=5).mean()
    df = df.dropna()

    # 2) features de lag e média móvel
    df["lag_1h"]  = df["energy_kWh"].shift(1)
    df["lag_24h"] = df["energy_kWh"].shift(24)
    df["lag_7d"]  = df["energy_kWh"].shift(24 * 7)
    df["avg_7d"]  = df["energy_kWh"].shift(1).rolling(window=24 * 7).mean()
    df = df.dropna()

    features = ["lag_1h", "lag_24h", "lag_7d", "avg_7d"]
    data   = df[features].values           # (T, 4)
    target = df["energy_kWh"].values       # (T,)

    # 3) criação de janelas look-back
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback : i, :])  # (lookback, 4)
        y.append(target[i])
    X = np.array(X).reshape(len(X), -1)      # (n_samples, lookback*4)
    y = np.array(y).reshape(-1, 1)           # (n_samples, 1)

    # 4) split temporal 80/20
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 5) padronização
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    X_train = scaler_X.transform(X_train)
    X_test  = scaler_X.transform(X_test)
    y_train = scaler_y.transform(y_train).ravel()
    y_test  = scaler_y.transform(y_test).ravel()

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y

best_params = {
    'C1': {'C': 1, 'epsilon': 0.1, 'gamma': 'auto'}, 
    'C2': {'C': 10, 'epsilon': 0.1, 'gamma': 'auto'}, 
    'C3': {'C': 10, 'epsilon': 0.1, 'gamma': 'scale'}, 
    'C4': {'C': 1, 'epsilon': 0.01, 'gamma': 'auto'}, 
    'C5': {'C': 1, 'epsilon': 0.1, 'gamma': 'auto'}
}

default_look_back = 12
num_clients = 5
seq_length = default_look_back

def make_sliding_windows(X, y, seq_length):
    y_flat = y.flatten()
    n_samples, n_features = X.shape
    n_windows = n_samples - seq_length
    Xw = np.zeros((n_windows, seq_length * n_features))
    yw = np.zeros(n_windows,)
    for idx in range(n_windows):
        window = X[idx: idx + seq_length]
        Xw[idx, :] = window.reshape(-1)
        yw[idx]    = y_flat[idx + seq_length]
    return Xw, yw


# Prepare metrics storage
metrics = []

for client_id in range(num_clients):
    key = f"C{client_id+1}"
    params = best_params[key]

    # Create client-specific folder
    client_folder = f"svm_predictions/smoothed/client_{client_id}"
    os.makedirs(client_folder, exist_ok=True)

    # Load data
    x_train, x_test, y_train, y_test, scaler_X, scaler_y = load_data(client_id, num_clients)

    # Window the data
    Xw_train, yw_train = make_sliding_windows(x_train, y_train, seq_length)
    Xw_test, yw_test   = make_sliding_windows(x_test,  y_test,  seq_length)

    # Fit model
    model = SVR(kernel='rbf', **params)
    model.fit(Xw_train, yw_train)

    # Test predictions and metrics
    y_pred_scaled = model.predict(Xw_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
    y_true = scaler_y.inverse_transform(yw_test.reshape(-1,1)).flatten()

    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    metrics.append({
        "Client":   key,
        "Test_MAE":  mae,
        "Test_MSE":  mse,
        "Test_RMSE": rmse,
        "Test_MAPE": mape,
        "Test_R2":   r2,
    })

    # Plot smoothed full
    plt.figure(figsize=(12,5))
    plt.plot(y_true, label='True', linewidth=2)
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.title(f'Client {client_id} – SVM Predictions')
    plt.xlabel('Window index')
    plt.ylabel('Energy (kWh)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{client_folder}/full_prediction.png")
    plt.close()

    # Plot smoothed zoom (first 100 windows)
    plt.figure(figsize=(12,5))
    plt.plot(y_true[:100], label='True', linewidth=2)
    plt.plot(y_pred[:100], label='Predicted', linestyle='--')
    plt.title(f'Client {client_id} – SVM Predictions (Zoomed)')
    plt.xlabel('Window index')
    plt.ylabel('Energy (kWh)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{client_folder}/zoomed_prediction.png")
    plt.close()

# Save aggregated metrics
pd.DataFrame(metrics).to_csv('metrics_per_client_SVM.csv', index=False)

print("✅ Training complete: metrics + plots saved.")



