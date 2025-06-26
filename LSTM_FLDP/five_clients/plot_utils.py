import os
import torch
from opacus_fl.load_data import load_data
from opacus_fl.task import LoadForecastingModel
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import argparse

CLIENT_IDS = {
    0: "Residential_4.csv",
    1: "Residential_8.csv",
    2: "Residential_9.csv",
    3: "Residential_10.csv",
    4: "Residential_13.csv",
}


def evaluate_client(
    model,
    client_id,
    num_partitions,
    batch_size,
    device,
    lookback: int = 12,
):
    # Load data with lookback windows
    _, test_loader, _, scaler_y = load_data(
        partition_id=client_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        shuffle=False,
        lookback=lookback,
    )

    model.eval()
    preds_scaled, trues_scaled = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            preds_scaled.append(preds.cpu().numpy())
            trues_scaled.append(y_batch.numpy())

    preds_scaled = np.vstack(preds_scaled)
    trues_scaled = np.vstack(trues_scaled)

    # Inverse transform
    preds = scaler_y.inverse_transform(preds_scaled)
    trues = scaler_y.inverse_transform(trues_scaled)

    # Metrics
    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(trues, preds)

    return trues, preds, {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


def plot_predictions(client_id, y_true, y_pred, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Full series
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="True", linewidth=2)
    plt.plot(y_pred, label="Predicted", linestyle="--")
    plt.title(f"Client {client_id} – Full Test Prediction")
    plt.xlabel("Time step")
    plt.ylabel("Energy (kWh)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"predictions_client_{client_id}.png"), dpi=300)
    plt.close()

    # Zoom first 100
    plt.figure(figsize=(12, 5))
    plt.plot(y_true[:100], label="True", linewidth=2)
    plt.plot(y_pred[:100], label="Predicted", linestyle="--")
    plt.title(f"Client {client_id} – Zoomed Test Prediction (First 100)")
    plt.xlabel("Time step")
    plt.ylabel("Energy (kWh)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"zoomed_predictions_client_{client_id}.png"), dpi=300)
    plt.close()


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model
    model = LoadForecastingModel(input_size=4, units=64, output_size=1, dropout=0.0)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    all_metrics = []
    for client_id in CLIENT_IDS:
        y_true, y_pred, metrics = evaluate_client(
            model=model,
            client_id=client_id,
            num_partitions=len(CLIENT_IDS),
            batch_size=args.batch_size,
            device=device,
            lookback=12
        )
       # plot_predictions(client_id, y_true, y_pred, args.output_dir)
        metrics["client_id"] = client_id
        all_metrics.append(metrics)

    df_metrics = pd.DataFrame(all_metrics).set_index("client_id")
    print("\nMatriz de métricas por cliente:\n", df_metrics)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "metrics_by_client.csv")
    df_metrics.to_csv(csv_path)
    print(f"\nMétricas salvas em {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Avalia modelo PyTorch por cliente, salva métricas e plota previsões"
    )
    parser.add_argument(
        "--model-path", default="final_model.pt",
        help="caminho para o state_dict do modelo salvo"
    )
    parser.add_argument(
        "--output-dir", default="evaluation_results",
        help="diretório de saída para gráficos e CSV"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="tamanho do batch para o DataLoader de teste"
    )
    args = parser.parse_args()
    main(args)
