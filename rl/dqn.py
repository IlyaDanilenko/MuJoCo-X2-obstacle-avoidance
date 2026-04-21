# rl/dqn.py - Double DQN в латенте (сетка vxxvy)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



def build_action_grid(
    vx_low: float,
    vx_high: float,
    n_vx_bins: int,
    vy_low: float,
    vy_high: float,
    n_vy_bins: int,
) -> np.ndarray:
    vx_centers = np.linspace(float(vx_low), float(vx_high), int(n_vx_bins), dtype=np.float32)
    vy_centers = np.linspace(float(vy_low), float(vy_high), int(n_vy_bins), dtype=np.float32)
    return np.array([(vx, vy) for vy in vy_centers for vx in vx_centers], dtype=np.float32)


class LatentQNetwork(nn.Module):
    """Вход: латент z [B, latent_dim]; выход: Q(z,·) для всех дискретных действий [B, n_actions]."""

    def __init__(self, latent_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_actions = n_actions
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.q_head = nn.Linear(hidden_dim, n_actions)
        nn.init.orthogonal_(self.q_head.weight, gain=0.01)
        nn.init.constant_(self.q_head.bias, 0.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.trunk(z)
        return self.q_head(h)


class LatentReplayBuffer:
    def __init__(self, capacity: int, latent_dim: int, rng: np.random.Generator):
        self.capacity = max(1, int(capacity))
        self.latent_dim = int(latent_dim)
        self.rng = rng
        self.z = np.zeros((self.capacity, self.latent_dim), dtype=np.float32)
        self.z_next = np.zeros((self.capacity, self.latent_dim), dtype=np.float32)
        self.a = np.zeros((self.capacity,), dtype=np.int64)
        self.r = np.zeros((self.capacity,), dtype=np.float32)
        self.done = np.zeros((self.capacity,), dtype=np.float32)
        self._idx = 0
        self.size = 0

    def add(self, z: np.ndarray, a: int, r: float, z_next: np.ndarray, done: float) -> None:
        i = self._idx
        self.z[i] = z
        self.z_next[i] = z_next
        self.a[i] = a
        self.r[i] = r
        self.done[i] = done
        self._idx = (self._idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = min(int(batch_size), self.size)
        idxs = self.rng.integers(0, self.size, size=n)
        return self.z[idxs], self.a[idxs], self.r[idxs], self.z_next[idxs], self.done[idxs]
