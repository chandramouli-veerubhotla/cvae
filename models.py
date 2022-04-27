import torch
import torch.nn as nn


def label_onehot(idx, n):
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    return onehot


class ConditionalVaeFFN(nn.Module):
    """Conditional VAE for MNIST."""

    def __init__(self, inp_size: int, hidden_size: int, latent_size: int, condition_size: int):
        super().__init__()

        self.latent_size = latent_size
        # Encoder
        self.encoder = Encoder(inp_size + condition_size, hidden_size, latent_size)
        self.decoder = Decoder(inp_size, hidden_size, latent_size + condition_size)

    def re_parameterize(self, mu, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x, label):
        means, var = self.encoder(x, label)
        z = self.re_parameterize(means, var)
        recon_x = self.decoder(z, label)

        return recon_x, means, var, z

    def generate(self, conditions):
        z = torch.randn([conditions.shape[0], self.latent_size]).to(conditions.device)  # Batch-Size x latent-size
        x = self.decoder(z, conditions)
        return x


class Encoder(nn.Module):

    def __init__(self, inp_size, hidden_size, latent_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(inp_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        # Mean and variance
        self.mean = nn.Linear(hidden_size, latent_size)
        self.var = nn.Linear(hidden_size, latent_size)

    def forward(self, x, label):
        inp = torch.cat((x, label), dim=-1)
        logits = self.net(inp)
        mean = self.mean(logits)
        var = self.var(logits)

        return mean, var


class Decoder(nn.Module):

    def __init__(self, inp_size, hidden_size, latent_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, inp_size),
            nn.Sigmoid()
        )

    def forward(self, z, label):
        inp = torch.cat((z, label), dim=-1)
        return self.net(inp)
