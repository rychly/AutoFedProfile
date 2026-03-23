import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader[Any],
    criterion: nn.Module,
    device: str | torch.device = "cpu",
) -> float:
    model.eval()
    model.to(device)
    criterion.to(device)

    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            # Handle both TensorDataset [batch] and standard (X, y) tuples
            inputs = (
                batch[0].to(device)
                if isinstance(batch, (list, tuple))
                else batch.to(device)
            )

            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size

    return total_loss / num_samples if num_samples > 0 else 0.0


def train_autoencoder(
    model: nn.Module,
    x_train: np.ndarray | torch.Tensor,
    x_val: np.ndarray | torch.Tensor,
    max_epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    criterion: nn.Module | None = None,
    early_stopping_patience: int = 10,
    device: str | torch.device = "cpu",
) -> dict[str, list[float]]:
    # Prepare DataLoaders (Equivalent to Keras x_train, x_train)
    t_train = torch.as_tensor(x_train, dtype=torch.float32)
    t_val = torch.as_tensor(x_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(t_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(t_val), batch_size=batch_size)

    # Setup Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_criterion = criterion or nn.BCELoss()
    model.to(device)
    loss_criterion.to(device)

    # Early Stopping Variables
    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    history = {"loss": [], "val_loss": []}

    for epoch in range(max_epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        for [batch] in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = loss_criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)

        # --- Validation Phase ---
        avg_train_loss = train_loss / len(t_train)
        avg_val_loss = evaluate_model(model, val_loader, loss_criterion, device)

        history["loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        # Log progress (Equivalent to verbose=1)
        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1}/{max_epochs} - loss: {avg_train_loss:.4f} - val_loss: {avg_val_loss:.4f}"
            )

        # --- Early Stopping Logic ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Restore best weights (Equivalent to restore_best_weights=True)
    model.load_state_dict(best_model_wts)
    return history
