import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def compute_recon_error(
    model: nn.Module,
    data_loader: np.ndarray | DataLoader[Any],
    criterion: nn.Module,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Calculates the specific criterion loss for every single sample in the loader."""
    model.eval()
    model.to(device)
    criterion.to(device)
    all_errors = []

    # Ensure the criterion does not average the results automatically
    # We need the 'none' reduction to get one value per sample
    original_reduction = getattr(criterion, "reduction", "mean")
    criterion.reduction = "none"  # pyright: ignore[reportArgumentType]

    iterator = (
        [torch.as_tensor(data_loader).float()]
        if isinstance(data_loader, np.ndarray)
        else data_loader
    )

    with torch.no_grad():
        for batch in iterator:
            inputs = (
                batch[0].to(device)
                if isinstance(batch, (list, tuple))
                else batch.to(device)
            )

            outputs = model(inputs)
            # This now returns a tensor of shape (batch_size,)
            # instead of a single averaged scalar
            loss = criterion(outputs, inputs)

            # If the loss is multidimensional (like MSE before reduction),
            # we average across the feature dimension (dim=1)
            if loss.dim() > 1:
                loss = torch.mean(loss, dim=1)

            all_errors.extend(loss.cpu().numpy())

    # Restore original reduction setting just in case the criterion is reused elsewhere
    criterion.reduction = original_reduction  # pyright: ignore[reportArgumentType]

    return np.array(all_errors)


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader[Any],
    criterion: nn.Module,
    device: str | torch.device = "cpu",
) -> float:
    """Standard evaluation that returns the mean loss across the dataset."""
    errors = compute_recon_error(model, data_loader, criterion, device)
    return float(np.mean(errors)) if len(errors) > 0 else 0.0


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
