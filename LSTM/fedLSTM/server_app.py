import logging
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedProx
from fedavg2.task import load_model, get_weights, set_weights

class SaveModelStrategy(FedProx):
    def __init__(
        self,
        *,
        total_rounds: int,
        proximal_mu: float = 0.1,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_available_clients: int = 2,
        initial_parameters,
    ):
        super().__init__(
            proximal_mu=proximal_mu,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
        )
        self.total_rounds = total_rounds

    def aggregate_fit(self, server_round: int, results, failures):

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if (
            aggregated_parameters is not None
            and server_round == self.total_rounds
        ):
            ndarrays = parameters_to_ndarrays(aggregated_parameters)

            model = load_model(
                input_size=4,
                units=64,
                output_size=1,
                dropout=0.2,
            )
            set_weights(model, ndarrays)
            torch.save(model.state_dict(), "final_model.pt")
            logging.info(f"Saved final_model.pt on round {server_round}")

        return aggregated_parameters, aggregated_metrics

def server_fn(context) -> ServerAppComponents:
    num_rounds = context.run_config["num-server-rounds"]
    local_epochs = context.run_config["local-epochs"]

    # Initialize a dummy model to extract its initial weights
    model = load_model(input_size=4, units=64, output_size=1, dropout=0.2)
    init_params = ndarrays_to_parameters(get_weights(model))

    # Use our custom strategy that only writes on the last round
    strategy = SaveModelStrategy(
        total_rounds=num_rounds,
        proximal_mu=0.1,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=init_params,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(config=config, strategy=strategy)


app = ServerApp(server_fn=server_fn)