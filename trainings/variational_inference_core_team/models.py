import torch
from torch import nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self, input_size=784, z_size=20):
        super().__init__()
        hidden_size = int((input_size - z_size) / 2 + z_size)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, z_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, x):
        x = x.view(-1, 784)
        z = self.encoder(x)
        x = self.decoder(z)

        if self.training:
            return x
        else:
            return F.sigmoid(x)


class VAE(nn.Module):
    def __init__(self, input_size=784, z_size=20):
        super().__init__()
        hidden_size = int((input_size - z_size) / 2 + z_size)
        self.z_size = z_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, z_size * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # unit gaussian
        z = mu + eps * std
        return z

    def forward(self, x):
        x = x.view(-1, 784)
        variational_params = self.encoder(x)
        mu = variational_params[..., :self.z_size]
        log_var = variational_params[..., self.z_size:]
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), z, mu, log_var
