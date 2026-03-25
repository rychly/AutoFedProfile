from collections import OrderedDict

import torch
from flwr.client import NumPyClient
from flwr.common import Scalar
from torch.utils.data import DataLoader, TensorDataset

from tls_profiling.autoencoder.models_torch import Net
from tls_profiling.autoencoder.train_torch import evaluate_model, train_autoencoder


class FlowerClient(NumPyClient):
    def __init__(
        self,
        data_partition,
        input_dim: int,
        encoding_dim: int,
        validation_ration: float,
        criterion: torch.nn.Module,
        device: str | torch.device = "cpu",
    ):
        self.device = device
        self.model = Net(input_dim=input_dim, encoding_dim=encoding_dim).to(device)
        self.criterion = criterion

        # Split node's local partition into train/val
        split_idx = int((1 - validation_ration) * len(data_partition))
        self.x_train = data_partition[:split_idx]
        self.x_val = data_partition[split_idx:]

        # x_train and x_val tensors should be copied by x_*.copy() to enable writing (not required) and fix Ray backend UserWarning: The given NumPy array is not writable ...
        # wont do that as there qould be memory overhead (sharing the tensors by default is mroe memory efficient)
        self.train_loader = DataLoader(
            TensorDataset(torch.as_tensor(self.x_train).float()),
            batch_size=16,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            TensorDataset(torch.as_tensor(self.x_val).float()), batch_size=16
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        _ = self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        history = train_autoencoder(
            self.model,
            self.x_train,
            self.x_val,
            max_epochs=int(config.get("local_epochs", 1)),
            batch_size=int(config.get("batch_size", 32)),
            lr=float(config.get("lr", 1e-3)),
            criterion=self.criterion,
            device=self.device,
        )
        last_loss = (
            history["loss"][-1] if "loss" in history and history["loss"] else 0.0
        )
        num_examples = len(self.x_train)
        metrics: dict[str, Scalar] = {"train_loss": float(last_loss)}
        return self.get_parameters(config={}), num_examples, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = evaluate_model(
            self.model, self.val_loader, criterion=self.criterion, device=self.device
        )
        num_examples = len(self.x_val)
        metrics: dict[str, Scalar] = {"mse": float(loss)}
        return loss, num_examples, metrics
