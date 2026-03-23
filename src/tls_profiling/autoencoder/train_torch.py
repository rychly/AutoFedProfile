import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train_autoencoder(
    model: nn.Module,
    x_train: np.ndarray | torch.Tensor,
    x_val: np.ndarray | torch.Tensor,
    max_epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    early_stopping_patience: int = 10,
    device: str | torch.device = "cpu",
) -> dict[str, list[float]]:
    # Prepare DataLoaders (Equivalent to Keras x_train, x_train)
    t_train = torch.tensor(x_train, dtype=torch.float32)
    t_val = torch.tensor(x_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(t_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(t_val), batch_size=batch_size)

    # Setup Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_criterion = nn.BCELoss()
    model.to(device)

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
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for [v_batch] in val_loader:
                v_batch = v_batch.to(device)
                v_outputs = model(v_batch)
                v_loss = loss_criterion(v_outputs, v_batch)
                val_loss += v_loss.item() * v_batch.size(0)

        avg_train_loss = train_loss / len(t_train)
        avg_val_loss = val_loss / len(t_val)

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
