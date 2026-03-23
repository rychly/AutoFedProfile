import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoding_dim: int,
        conv_input_size: int = 20,
        intermediate_dim: int = 64,
        activation_fn=F.relu,
        output_activation_fn=torch.sigmoid,
    ):
        super(Net, self).__init__()

        if conv_input_size <= 0 or conv_input_size > input_dim:
            raise ValueError(
                f"conv_input_size must be in [1, {input_dim}], got {conv_input_size}"
            )

        self.conv_input_size = conv_input_size
        self.activation_fn = activation_fn
        self.output_activation_fn = output_activation_fn

        # --- Encoder Branches ---
        # Conv Branch: Input (Batch, 1, conv_input_size)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Calculate Flatten size after Conv/Pool
        # Formula: L_out = floor((L_in - kernel_size) / stride + 1)
        conv_out_size = (conv_input_size - 3) // 1 + 1
        pool_out_size = conv_out_size // 2
        self.flat_size = 32 * pool_out_size

        # Dense Branch: Remaining features
        self.remaining_dim = input_dim - conv_input_size
        self.dense_branch = nn.Linear(self.remaining_dim, intermediate_dim)

        # --- Shared Encoder Hidden Layers ---
        self.enc_hidden = nn.Linear(self.flat_size + intermediate_dim, intermediate_dim)
        self.latent = nn.Linear(intermediate_dim, encoding_dim)

        # --- Decoder ---
        self.dec_hidden = nn.Linear(encoding_dim, intermediate_dim)
        self.recon = nn.Linear(intermediate_dim, input_dim)

        # Apply Xavier Uniform initialization to match Keras defaults
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        # Slice 1: Convolutional part
        # Keras: (batch, length) -> (batch, length, 1)
        # PyTorch: (batch, length) -> (batch, 1, length)
        x_conv = x[:, : self.conv_input_size].unsqueeze(1)
        x_conv = self.activation_fn(self.conv1(x_conv))
        x_conv = self.pool1(x_conv)
        x_conv = torch.flatten(x_conv, start_dim=1)

        # Slice 2: Dense part
        x_dense = x[:, self.conv_input_size :]
        x_dense = self.activation_fn(self.dense_branch(x_dense))

        # Combine and Bottleneck
        combined = torch.cat([x_conv, x_dense], dim=1)
        hidden = self.activation_fn(self.enc_hidden(combined))
        return self.activation_fn(self.latent(hidden))

    def decode(self, z):
        hidden_dec = self.activation_fn(self.dec_hidden(z))
        return self.output_activation_fn(self.recon(hidden_dec))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
