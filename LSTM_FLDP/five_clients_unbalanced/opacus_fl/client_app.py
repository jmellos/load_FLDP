import warnings
import torch
from opacus import PrivacyEngine
from opacus_fl.task import LoadForecastingModel, get_weights, set_weights, test, train
from opacus_fl.load_data import load_data
import logging

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

warnings.filterwarnings("ignore", category=UserWarning)


class FlowerClient(NumPyClient):
    def __init__(
        self,
        train_loader,
        test_loader,
        target_delta,
        max_grad_norm,
        local_epochs,
    ) -> None:
        super().__init__()
        self.model = LoadForecastingModel(
            input_size=4,
            units=64,
            output_size=1,
            dropout=0.0,
        )
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.max_grad_norm = max_grad_norm
        self.local_epochs = local_epochs
        self.target_delta = target_delta
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = train_loader.batch_size

    def fit(self, parameters, config):
        set_weights(self.model, parameters)

        noise_multiplier = config["noise_multiplier"]

        lr = config.get("learning-rate", 1e-3)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
        )

        privacy_engine = PrivacyEngine()
        self.model, optimizer, train_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=optimizer,
            data_loader=self.train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )

        epsilon = train(
            self.model,
            self.train_loader,
            privacy_engine,
            optimizer,
            self.target_delta,
            device=self.device,
            epochs=self.local_epochs,
        )

        if epsilon is not None:
            print(f"Epsilon value for delta={self.target_delta} is {epsilon:.2f}")
        else:
            print("Epsilon value not available.")

        return (
            get_weights(self.model),
            len(self.train_loader.dataset),
            {"dataset_size": len(self.train_loader.dataset)},
        )

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        self.model.to(self.device)
        mse, mae = test(self.model, self.test_loader, self.device)
        return float(mse), len(self.test_loader.dataset), {"mae": float(mae)}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    train_loader, test_loader, _, _ = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions
    )
    return FlowerClient(
        train_loader,
        test_loader,
        context.run_config["target-delta"],
        context.run_config["max-grad-norm"],
        context.run_config["epochs"],
    ).to_client()


app = ClientApp(client_fn=client_fn)
