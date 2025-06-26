import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int = 32,
    shuffle: bool = False,
    lookback: int = 12,
):
    # 1) mapa de arquivos por cliente
    client_files = {
        0: "Residential_4.csv",
        1: "Residential_8.csv",
        2: "Residential_9.csv",
        3: "Residential_10.csv",
        4: "Residential_13.csv",
    }
    csv_file_path = client_files[partition_id]
    df = pd.read_csv(csv_file_path)

    # 2) interpolação e suavização
    if df['energy_kWh'].isnull().any():
        df['energy_kWh'] = df['energy_kWh'].interpolate()
    df['energy_kWh'] = df['energy_kWh'].rolling(window=5).mean()
    df = df.dropna()

    # 3) features de lag e médias móveis
    df['lag_1h']  = df['energy_kWh'].shift(1)
    df['lag_24h'] = df['energy_kWh'].shift(24)
    df['lag_7d']  = df['energy_kWh'].shift(24*7)
    df['avg_7d']  = df['energy_kWh'].shift(1).rolling(window=24*7).mean()
    df = df.dropna()

    features = ['lag_1h', 'lag_24h', 'lag_7d', 'avg_7d']
    data = df[features].values  # shape (T, 4)

    # 4) geração de janelas look-back
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback : i, :])  # (lookback, num_features)
        y.append(data[i, 0])                 # target: energy_kWh via lag_1h
    X = np.array(X)       # (n_samples, lookback, num_features)
    y = np.array(y).reshape(-1, 1)

    # 5) split temporal 80/20
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 6) padronização (scaler aplicado no 2D e remontado para 3D)
    n_train, lb, nf = X_train.shape
    X_train_flat = X_train.reshape(n_train, lb * nf)
    scaler_X = StandardScaler().fit(X_train_flat)
    X_train_flat = scaler_X.transform(X_train_flat)
    X_train = X_train_flat.reshape(n_train, lb, nf)

    n_test = X_test.shape[0]
    X_test_flat = X_test.reshape(n_test, lb * nf)
    X_test_flat = scaler_X.transform(X_test_flat)
    X_test = X_test_flat.reshape(n_test, lb, nf)

    scaler_y = StandardScaler().fit(y_train)
    y_train = scaler_y.transform(y_train)
    y_test  = scaler_y.transform(y_test)

    # 7) tensores e DataLoaders
    tX_train = torch.tensor(X_train, dtype=torch.float32)
    ty_train = torch.tensor(y_train, dtype=torch.float32)
    tX_test  = torch.tensor(X_test,  dtype=torch.float32)
    ty_test  = torch.tensor(y_test,  dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(tX_train, ty_train),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    test_loader = DataLoader(
        TensorDataset(tX_test, ty_test),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader, scaler_X, scaler_y
