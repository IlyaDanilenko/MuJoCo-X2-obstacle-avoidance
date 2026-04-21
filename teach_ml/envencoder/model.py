# teach_ml/envencoder/model.py - VAE envencoder (depth+odom --> латент)

import torch
import torch.nn as nn
import torch.nn.functional as F


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class EnvEncoder(nn.Module):
    """CNN по depth + MLP по imu --> μ, log σ² размерности latent_dim."""

    def __init__(self, latent_dim: int = 256, imu_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.imu_dim = imu_dim
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 224 --> 112 --> 56 --> 28 --> 14 --> 7
        self.fc_vis = nn.Linear(512 * 7 * 7, 512)
        self.fc_imu = nn.Linear(imu_dim, 64)
        self.fc_mu = nn.Linear(512 + 64, latent_dim)
        self.fc_logvar = nn.Linear(512 + 64, latent_dim)

    def forward(self, depth: torch.Tensor, imu: torch.Tensor):
        """
        depth: [B, 1, H, W] в [0, 1]
        imu: [B, imu_dim]
        """
        h = self.conv(depth)
        h = h.reshape(h.size(0), -1)
        h = F.leaky_relu(self.fc_vis(h), 0.2)
        hi = F.leaky_relu(self.fc_imu(imu), 0.2)
        x = torch.cat([h, hi], dim=1)
        mu = self.fc_mu(x)
        # log σ² ∈ [-20, 2]: защита от exp(logvar)-->inf в KL и в репараметризации (типичный VAE-трюк)
        logvar = self.fc_logvar(x).clamp(-20.0, 2.0)
        return mu, logvar


class EnvDecoder(nn.Module):
    """z --> depth [0,1] + imu."""

    def __init__(self, latent_dim: int = 256, imu_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.imu_dim = imu_dim
        self.fc = nn.Linear(latent_dim, 512 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid(),
        )
        self.imu_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, imu_dim),
        )

    def forward(self, z: torch.Tensor):
        h = self.fc(z)
        h = h.reshape(-1, 512, 7, 7)
        depth = self.deconv(h)
        imu = self.imu_head(z)
        return depth, imu


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0,I)); logvar должен быть ограничен (см. EnvEncoder)."""
    lv = torch.clamp(logvar, -20.0, 2.0)
    return -0.5 * torch.mean(1.0 + lv - mu.pow(2) - lv.exp())
