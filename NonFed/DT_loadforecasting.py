import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

def load_data(partition_id, num_partitions):
    client_files = {
        0: "Residential_4.csv",
        1: "Residential_8.csv",
        2: "Residential_9.csv",
        3: "Residential_10.csv",
        4: "Residential_13.csv",
    }
    csv_file_path = client_files[partition_id]
    df = pd.read_csv(csv_file_path)


    if df['energy_kWh'].isnull().any():
        df['energy_kWh'] = df['energy_kWh'].interpolate(method='linear')
    df['energy_kWh'] = df['energy_kWh'].rolling(window=5).mean()
    df = df.dropna()

    # 3) criação de features de lag e médias móveis
    df['lag_1h']  = df['energy_kWh'].shift(1)
    df['lag_24h'] = df['energy_kWh'].shift(24)
    df['lag_7d']  = df['energy_kWh'].shift(24*7)
    df['avg_7d']  = df['energy_kWh'].shift(1).rolling(window=24*7).mean()
    df = df.dropna()

    # 4) inputs e target
    X = df[['lag_1h', 'lag_24h', 'lag_7d', 'avg_7d']].values
    y = df['energy_kWh'].values

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test

# Parameters
num_clients = 5
os.makedirs("dt_predictions", exist_ok=True)
metrics_smoothed = []

for client_id in range(num_clients):
    os.makedirs(f"dt_predictions/client_{client_id}", exist_ok=True)

    x_train, x_test, y_train, y_test = load_data(client_id, num_clients)

    # Train with best hyperparameters
    dt_model = DecisionTreeRegressor(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features=None,
        criterion='squared_error',
        random_state=42
    )
    dt_model.fit(x_train, y_train)

    # Predictions
    y_pred = dt_model.predict(x_test)

    # Metrics
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse= np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    metrics_smoothed.append({
        "Client": f"C{client_id+1}",
        "MAE": mae, "MSE": mse, "RMSE": rmse,
        "MAPE": mape, "R2": r2
    })
    print(f"  MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    # Plot full
    plt.figure(figsize=(12,5))
    plt.plot(y_test, label='True', linewidth=2)
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.title(f'Client {client_id} – DT Full')
    plt.xlabel('Time step'); plt.ylabel('Energy (kWh)')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"dt_predictions/client_{client_id}/full_prediction.png")
    plt.close()

    # Plot zoom
    plt.figure(figsize=(12,5))
    plt.plot(y_test[:100], label='True', linewidth=2)
    plt.plot(y_pred[:100], label='Predicted', linestyle='--')
    plt.title(f'Client {client_id} – DT Zoom')
    plt.xlabel('Time step'); plt.ylabel('Energy (kWh)')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"dt_predictions/client_{client_id}/zoomed_prediction.png")
    plt.close()

pd.DataFrame(metrics_smoothed).to_csv('metrics_per_client_DT_smoothed.csv', index=False)
print("✅ DT done with tuned hyperparameters: metrics + plots saved.")
