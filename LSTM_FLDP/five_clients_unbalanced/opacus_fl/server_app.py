"""opacus: Training with Sample-Level Differential Privacy using Opacus Privacy Engine."""

import logging
from typing import Dict, List, Tuple
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, FitIns, Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedProx
from opacus.accountants.utils import get_noise_multiplier
from opacus_fl.task import LoadForecastingModel, get_weights, set_weights
import torch

# Opacus logger seems to change the flwr logger to DEBUG level. Set back to INFO
logging.getLogger("flwr").setLevel(logging.INFO)

client_dataset_sizes: Dict[str, int] = {}


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
        config: Dict[str, Scalar],
        **kwargs,
    ):
        self.total_rounds = total_rounds
        self.config = config
        
        super().__init__(
            proximal_mu=proximal_mu,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
            **kwargs,
        )

    def configure_fit(
        self,
        server_round: int,
        parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        
        sampled_clients = client_manager.sample(
            num_clients=int(self.fraction_fit * len(client_manager.all().values())),
            min_num_clients=self.min_fit_clients
        ) 

        total_size = sum(client_dataset_sizes.get(c.cid, 1) for c in sampled_clients)

        fit_ins_list = []
        for client in sampled_clients:
            n_k = client_dataset_sizes.get(client.cid, 1)
            q_k = n_k / total_size

            # Sensitivity adjusted by FedProx effect
            prox_mu = self.config.get("prox_mu", 0.1)
            adjusted_clip_norm = self.config["max_grad_norm"] / (1 + prox_mu)

            # Calculate custom noise
            noise_multiplier = get_noise_multiplier(
                target_epsilon=self.config["target_epsilon"],
                target_delta=self.config["target_delta"],
                sample_rate=q_k,
                steps=self.config["num_server_rounds"],
                accountant="rdp",
            )

            print(f"[Server] Client {client.cid} - noise_multiplier: {noise_multiplier:.4f}")

            fit_config = {
                "epochs": self.config["epochs"],
                "noise_multiplier": noise_multiplier,
                "max_grad_norm": self.config["max_grad_norm"],
                "prox_mu": prox_mu,
                "dataset_size": n_k,
            }

            fit_ins_list.append((client, FitIns(parameters, fit_config)))

        return fit_ins_list
    
    def aggregate_fit(self, server_round, results, failures):

        for client, fit_res in results:
            size = fit_res.metrics.get("dataset_size")
            if size is not None:
                client_dataset_sizes[client.cid] = int(size)

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if (aggregated_parameters is not None and server_round == self.total_rounds):
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            model = LoadForecastingModel(
                input_size=4,  
                units=64,
                output_size=1,
                dropout=0.0,
            )
            set_weights(model, ndarrays)
            torch.save(model.state_dict(), "final_model.pt")

        return aggregated_parameters, aggregated_metrics


def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = context.run_config["num-server-rounds"]
    epochs = context.run_config["epochs"]
    min_available_clients = context.run_config["min-available-clients"]
    max_grad_norm = context.run_config["max-grad-norm"]
    target_epsilon = context.run_config["target-epsilon"]
    target_delta = context.run_config["target-delta"]
    prox_mu = context.run_config["prox_mu"]
    fraction_fit = context.run_config["fraction-fit"]

    init_model = LoadForecastingModel(input_size=4, units=64, output_size=1, dropout=0.2)
    init_params = ndarrays_to_parameters(get_weights(init_model))
                        

    strategy = SaveModelStrategy(
        total_rounds=num_rounds,
        proximal_mu=prox_mu,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=min_available_clients,
        initial_parameters=init_params,
        config={
            "epochs": epochs,
            "max_grad_norm": max_grad_norm,
            "target_epsilon": target_epsilon,
            "target_delta": target_delta,
            "num_server_rounds": num_rounds,
        },
    )

    config = ServerConfig(num_rounds=num_rounds)


    return ServerAppComponents(config=config, strategy=strategy)


app = ServerApp(server_fn=server_fn)
