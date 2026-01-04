import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def feature_masking(x, mask_prob=0.2):
    """
    Randomly masks features by setting them to zero.
    x: (batch, features)
    """
    mask = torch.rand_like(x) > mask_prob
    return x * mask

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


class AlignmentUniformityLoss(nn.Module):
    """
    Alignment and Uniformity losses for contrastive learning.

    want to infer the *common cause* from co-ocurring samples.

    contrastive loss is a lower bound on mutual information of embeddings

    embeddings are deterministic functions of samples

    so contrastive loss minimization maximizes mutual information
    between samples.

    In practice, this consists of masking a random subset of both
    arrays, and rewarding both the reconstruction loss of the sample
    as well as the proximity of the learned latent embedding.

    Authors mathematically show that there is a sweet spot in the mutual
    information between two unmasked "views" of a sample that containing
    the best representation.

    Even if the downstream task requiring semantic embeddings is not
    known, a task generally requires an unknown number of bits of
    mutual information in order to be sufficiently represented.
    Views that share approximately this amount of mutual information.

    The representation should be **minimal and sufficient**

    From *Understanding Contrastive Representation Learning through
    Alighnment and Uniformity on the Hypersphere*, (Wang & Isola, 2020)
    define metrics for alignment and uniformity.

    **alignment**: expected true pair feature distance. Measures distance
    between embeddings.

    **uniformity**: log expected gaussian potential of data pair. In
    other words, the expected alignment between those embeddings.
    Minimized when embeddings map to a gaussian distribution.

    Mutually minimizing uniformally and maximizing alignment encourages
    a semantically rich and gaussian-approximate latent embedding space.

    Paper shows examples where pareto front between the two metrics

    Based on "Understanding Contrastive Representation Learning through
    Alignment and Uniformity on the Hypersphere" (Wang & Isola, 2020)

    :@param alpha: alignment loss strength
    :@param t: uniformity loss temperature
    """
    def __init__(self, alpha=2, t=2):
        super().__init__()
        self.alpha = alpha
        self.t = t

    def alignment_loss(self, z1, z2):
        """
        Measures how close positive pairs are.
        Lower is better (pairs should be aligned).

        :@param z1: normed embeddings with shape (B, L)
        :@param z2: normed embeddings with shape (B, L)
        """
        # Compute pairwise distances between positive pairs
        return (z1 - z2).norm(dim=1, p=2).pow(self.alpha).mean()

    def uniformity_loss(self, z):
        """
        Measures how uniformly embeddings are distributed on the hypersphere.
        More negative is better (embeddings should be uniformly distributed).

        Args:
            z: normalized embeddings of shape (batch_size, embedding_dim)
        """
        # Compute pairwise distances for all pairs
        sq_pdist = torch.pdist(z, p=2).pow(2)
        # Uniformity: log of average of exp(-t * distances)
        return sq_pdist.mul(-self.t).exp().mean().log()

    def forward(self, z1, z2, lambda_align=1.0, lambda_uniform=1.0):
        """
        Combined alignment and uniformity loss.

        Args:
            z1, z2: normalized embeddings (batch_size, embedding_dim)
            lambda_align: weight for alignment loss
            lambda_uniform: weight for uniformity loss
        """
        align_loss = self.alignment_loss(z1, z2)
        uniform_loss = self.uniformity_loss(torch.cat([z1, z2], dim=0))

        total_loss = lambda_align * align_loss + lambda_uniform * uniform_loss

        return total_loss, align_loss, uniform_loss


class ContrastiveAutoencoder(nn.Module):
    """
    Simple contrastive autoencoder with encoder and decoder.
    """
    def __init__(self, input_dim, hidden_dims=[256, 128], latent_dim=64):
        """

        :@param input_dim:
        :@param nidden_dims: list of node counts for each hidden layer
        """
        super().__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x, normalize=True):
        """Encode input to latent space."""
        z = self.encoder(x)
        if normalize:
            z = F.normalize(z, dim=1, p=2)  # L2 normalize for hypersphere
        return z

    def decode(self, z):
        """Decode latent representation back to input space."""
        return self.decoder(z)

    def forward(self, x, normalize=True):
        """Full forward pass through encoder and decoder."""
        z = self.encode(x, normalize=normalize)
        x_recon = self.decode(z)
        return z, x_recon


class ContrastiveDataset(Dataset):
    """
    Dataset that generates pairs of augmented views of the same sample.
    For high-dimensional vectors, augmentations include:

    - Adding Gaussian noise
    - Dropout (randomly zeroing features)
    - Scaling
    """
    def __init__(self, data, noise_std=0.1, dropout_prob=0.1,
            scale_range=(0.9, 1.1)):
        """
        Args:
            data: numpy array of shape (n_samples, n_features)
            noise_std: standard deviation of Gaussian noise
            dropout_prob: probability of zeroing each feature
            scale_range: (min, max) range for random scaling
        """
        self.data = torch.FloatTensor(data)
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.scale_range = scale_range

    def augment(self, x):
        """Apply random augmentations to create a view."""
        # Add Gaussian noise
        noise = torch.randn_like(x) * self.noise_std
        x_aug = x + noise

        # Random dropout
        mask = torch.rand_like(x) > self.dropout_prob
        x_aug = x_aug * mask

        # Random scaling
        scale = torch.rand(1).item() \
                * (self.scale_range[1] - self.scale_range[0]) \
                + self.scale_range[0]
        x_aug = x_aug * scale

        return x_aug

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        # Create two augmented views of the same sample
        x1 = self.augment(x)
        x2 = self.augment(x)
        return x1, x2, x  # Return both views and original

def train_dae(model, dataloader, optimizer, device, mask_prob=0.2, epochs=50):
    """
    DenoisingAutoencoder training loop
    """
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

def train_cae(model, dataloader, optimizer, loss_func, recon_weight=0.1,
        device="cpu"):
    """
    ContrastiveAutoencoder training loop
    """
    model.train()
    total_loss = 0
    total_align = 0
    total_uniform = 0
    total_recon = 0

    for x1, x2, x_orig in dataloader:
        x1, x2, x_orig = x1.to(device), x2.to(device), x_orig.to(device)

        optimizer.zero_grad()

        # Encode both views (normalized for contrastive loss)
        z1 = model.encode(x1, normalize=True)
        z2 = model.encode(x2, normalize=True)

        # Contrastive loss (alignment + uniformity)
        contrast_loss, align_loss, uniform_loss = loss_func(z1, z2)

        # Reconstruction loss (optional, helps with stability)
        x1_recon = model.decode(z1)
        recon_loss = F.mse_loss(x1_recon, x_orig)

        # Combined loss
        loss = contrast_loss + recon_weight * recon_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_align += align_loss.item()
        total_uniform += uniform_loss.item()
        total_recon += recon_loss.item()

    n = len(dataloader)
    return total_loss/n, total_align/n, total_uniform/n, total_recon/n

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 500
    latent_dim = 64
    batch_size = 256
    n_epochs = 50
    lr = 1e-3

    ## generate synthetic data (replace with your actual data)
    n_samples = 10000
    data = np.random.randn(n_samples, input_dim).astype(np.float32)

    ## create dataset and dataloader
    dataset = ContrastiveDataset(data, noise_std=0.1, dropout_prob=0.1)
    dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
            )

    ## initialize model, loss, and optimizer
    model = ContrastiveAutoencoder(
        input_dim=input_dim,
        hidden_dims=[256, 128],
        latent_dim=latent_dim
        ).to(device)

    loss_func = AlignmentUniformityLoss(alpha=2, t=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=n_epochs,
            )
    # Training loop
    print(f"Training on {device}")
    print(f"Data shape: {data.shape}, Latent dim: {latent_dim}")
    print("-" * 70)

    for epoch in range(n_epochs):
        loss, align, uniform, recon = train_epoch(
            model, dataloader, optimizer, criterion,
            recon_weight=0.1, device=device
        )
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {loss:.4f} | "
                  f"Align: {align:.4f} | Uniform: {uniform:.4f} | Recon: {recon:.4f}")

    print("-" * 70)
    print("Training complete!")

    # Extract embeddings for entire dataset
    model.eval()
    with torch.no_grad():
        data_tensor = torch.FloatTensor(data).to(device)
        embeddings = model.encode(data_tensor, normalize=True).cpu().numpy()

    print(f"Final embeddings shape: {embeddings.shape}")jjkk
