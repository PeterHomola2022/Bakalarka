import flwr as fl
from flwr.common import Metrics
import numpy as np
from flwr.common import Config, NDArrays, Scalar, FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
from typing import List, Tuple, Union, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import warnings

import glob
import os
#
warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

config_dict = {
    "dropout": True,        # str key, bool value
    "learning_rate": 0.01,  # str key, float value
    "batch_size": 32,       # str key, int value
    "optimizer": "sgd",     # str key, str value
}

def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "batch_size": 32,
        "current_round": server_round,
        "local_epochs": 1 if server_round < 2 else 2,       # change local_epochs
    }
    return config

######

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)
            # Save the model
            torch.save(net.state_dict(), f"model_round_{3}.pth")     #server_round


        return aggregated_parameters, aggregated_metrics

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

net = Net().to(DEVICE)
#####
list_of_files = [fname for fname in glob.glob("./model_round_*")]
if len(list_of_files) !=0:
    latest_round_file = max(list_of_files, key=os.path.getctime)
    print("Loading pre-trained model from: ", latest_round_file)
    state_dict = torch.load(latest_round_file)
    net.load_state_dict(state_dict)





# Create strategy and run server
strategy = SaveModelStrategy(evaluate_metrics_aggregation_fn=weighted_average,
                                     on_fit_config_fn=fit_config,  # The fit_config function we defined earlier
                                     min_fit_clients=2, # Minimum number of clients to be sampled for the next round
                                     min_available_clients=2,  # Minimum number of clients that need to be connected to the server before a training round can start

    # (same arguments as FedAvg here)
)

# Define strategy
# strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average,
#                                     on_fit_config_fn=fit_config,  # The fit_config function we defined earlier
#                                     min_fit_clients=2, # Minimum number of clients to be sampled for the next round
#                                     min_available_clients=2,  # Minimum number of clients that need to be connected to the server before a training round can start
#                                     )

# list_of_files = [fname for fname in glob.glob("./model_round_*")]
# latest_round_file = max(list_of_files, key=os.path.getctime)
# print("Loading pre-trained model from: ", latest_round_file)
# state_dict = torch.load(latest_round_file)
# #help print
# print("Loading latest state dist was sucssessfull !!!")
# net.load_state_dict(state_dict)

# Start Flower server
fl.server.start_server(
    server_address="147.232.183.78:8080",          # v skole 147.232.156.185 # intrák 147.232.183.78
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)

#
# server
# musí ukladať model aby posielal clientovi váhy
# zároveň musí mať model aby vedel uložiť váhy
#
#
# client
# musí načítať váhy ktoré su uložene