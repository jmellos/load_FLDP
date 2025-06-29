import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
)
import matplotlib.pyplot as plt
import os

def load_data(partition_id: int, lookback: int = 12, test_size: float = 0.2):
    client_files = {
        0: "Residential_4.csv",
        1: "Residential_8.csv",
        2: "Residential_9.csv",
        3: "Residential_10.csv",
        4: "Residential_13.csv",
    }
    df = pd.read_csv(client_files[partition_id])
    df['energy_kWh'] = df['energy_kWh'].interpolate(method='linear')
    df['energy_kWh'] = df['energy_kWh'].rolling(window=5).mean()
    df = df.dropna()

    df['lag_1h']  = df['energy_kWh'].shift(1)
    df['lag_24h'] = df['energy_kWh'].shift(24)
    df['lag_7d']  = df['energy_kWh'].shift(24 * 7)
    df['avg_7d']  = df['energy_kWh'].shift(1).rolling(window=24 * 7).mean()
    df = df.dropna()

    features = ['lag_1h', 'lag_24h', 'lag_7d', 'avg_7d']
    data = df[features].values
    target = df['energy_kWh'].values

    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, :])
        y.append(target[i])
    X = np.array(X).reshape(len(X), -1)
    y = np.array(y).reshape(-1, 1)

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    X_train = scaler_X.transform(X_train)
    X_test  = scaler_X.transform(X_test)
    y_train = scaler_y.transform(y_train).ravel()
    y_test  = scaler_y.transform(y_test).ravel()

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y

# Parâmetros
num_clients = 5
lookback = 12
param_grid = {
    'alpha': [0.01, 0.1, 1.0, 10.0],
    'l1_ratio': [0.2, 0.5, 0.8]
}

metrics = []
os.makedirs("elasticnet_predictions", exist_ok=True)

for client_id in range(num_clients):
    os.makedirs(f"elasticnet_predictions/client_{client_id}", exist_ok=True)

    X_train, X_test, y_train, y_test, scaler_X, scaler_y = load_data(client_id, lookback)

    model = GridSearchCV(
        ElasticNet(max_iter=10000),
        param_grid,
        scoring='r2',
        cv=3,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    best_model = model.best_estimator_

    y_pred_scaled = best_model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    metrics.append({
        "Client": f"C{client_id+1}",
        "MAE": mae, "MSE": mse, "RMSE": rmse,
        "MAPE": mape, "R2": r2,
        "Best Params": model.best_params_
    })
    print(f"  MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    # Plot
    for tag, yp, yt in [("full", y_pred, y_true), ("zoom", y_pred[:100], y_true[:100])]:
        plt.figure(figsize=(12,5))
        plt.plot(yt, label="True", linewidth=2)
        plt.plot(yp, '--', label="Predicted")
        plt.title(f"Client {client_id} – ElasticNet ({tag})")
        plt.xlabel("Time step")
        plt.ylabel("Energy (kWh)")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(f"elasticnet_predictions/client_{client_id}/{tag}_prediction.png")
        plt.close()

# Export CSV com métricas
pd.DataFrame(metrics).to_csv("metrics_per_client_LR.csv", index=False)
print("✅ ElasticNet finalizado. Resultados salvos com sucesso.")
