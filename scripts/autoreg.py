import torch

def feature_masking(x, mask_prob=0.2):
    """
    Randomly masks features by setting them to zero.
    x: (batch, features)
    """
    mask = torch.rand_like(x) > mask_prob
    return x * mask
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.norm(x + residual)


class DenoisingAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim=1000,
        latent_dim=64,
        hidden_dims=(512, 256),
        num_res_blocks=2,
    ):
        super().__init__()

        # ----- Encoder -----
        encoder_layers = []
        dims = [input_dim] + list(hidden_dims)

        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.LayerNorm(dims[i + 1]))

        for _ in range(num_res_blocks):
            encoder_layers.append(ResidualBlock(hidden_dims[-1]))

        encoder_layers.append(nn.Linear(hidden_dims[-1], latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # ----- Decoder -----
        decoder_layers = []
        dims = [latent_dim] + list(reversed(hidden_dims)) + [input_dim]

        for i in range(len(dims) - 2):
            decoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.LayerNorm(dims[i + 1]))

        decoder_layers.append(nn.Linear(dims[-2], dims[-1]))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

def train_dae(
    model,
    dataloader,
    optimizer,
    device,
    mask_prob=0.2,
    epochs=50,
):
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for x in dataloader:
            x = x.to(device)

            x_corrupt = feature_masking(x, mask_prob)

            x_hat, _ = model(x_corrupt)
            loss = F.mse_loss(x_hat, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1:03d} | MSE: {avg_loss:.6f}")

model.eval()

with torch.no_grad():
    z = model.encoder(x)   # (batch, latent_dim)


