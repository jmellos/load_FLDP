import logging
import torch
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
from fedavg2.task import load_model, get_weights, set_weights, train, test
from fedavg2.load_data import load_data

class FlowerClient(NumPyClient):
    def __init__(
        self,
        train_loader,
        test_loader,
        learning_rate,
        local_epochs
    ) -> None:
        super().__init__()
        # Pega uma batch para inferir seq_len e num_features
        batch_X, _ = next(iter(train_loader))
        _, seq_len, num_features = batch_X.shape

        # Instancia o modelo com input_size correto
        self.model = load_model(
            input_size=num_features,
            units=64,
            output_size=1,
            dropout=0.0,
        )

        self.train_loader  = train_loader
        self.test_loader   = test_loader
        self.local_epochs  = local_epochs
        self.device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size    = train_loader.batch_size
        self.learning_rate = learning_rate

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        lr = config.get("learning-rate", 1e-3)
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=lr,
            weight_decay=1e-5
        )
        train(self.model, self.train_loader, optimizer, device=self.device, epochs=self.local_epochs)
        return get_weights(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        self.model.to(self.device)
        mse, mae = test(self.model, self.test_loader, self.device)
        return float(mse), len(self.test_loader.dataset), {"mae": float(mae)}

def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    lr = context.run_config.get("learning-rate", 1e-3)
    local_epochs = context.run_config["local-epochs"]

    train_loader, test_loader, _, _ = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=context.run_config.get("batch-size", 32),
        shuffle=False,
    )
    return FlowerClient(train_loader, test_loader, lr, local_epochs).to_client()


app = ClientApp(client_fn=client_fn)
