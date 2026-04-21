# rl/ppo.py - PPO в латенте: тренер и сетка действий vxxvy

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl.envencoder_bridge import RLEnvEncoderBridge


def _build_action_grid(
    vx_low: float,
    vx_high: float,
    n_vx_bins: int,
    vy_low: float,
    vy_high: float,
    n_vy_bins: int,
) -> np.ndarray:
    vx_centers = np.linspace(float(vx_low), float(vx_high), int(n_vx_bins), dtype=np.float32)
    vy_centers = np.linspace(float(vy_low), float(vy_high), int(n_vy_bins), dtype=np.float32)
    grid = np.array([(vx, vy) for vy in vy_centers for vx in vx_centers], dtype=np.float32)
    return grid


class LatentPolicyNetwork(nn.Module):
    """Вход: латент z [B, latent_dim]; выход: V(латент), logits [B, n_actions] по плоской сетке vxxvy."""

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
        self.value_head = nn.Linear(hidden_dim, 1)
        self.prior_head = nn.Linear(hidden_dim, n_actions)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)
        nn.init.orthogonal_(self.prior_head.weight, gain=0.01)
        nn.init.constant_(self.prior_head.bias, 0.0)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(z)
        v = self.value_head(h).squeeze(-1)
        logits = self.prior_head(h)
        return v, logits

    def value_only(self, z: torch.Tensor) -> torch.Tensor:
        v, _ = self.forward(z)
        return v


class PPOLatentTrainer:
    """
    PPO в латентном пространстве envencoder.
    Политика дискретная по 2D сетке (vx, vy); value — скаляр V(z).
    """

    def __init__(
        self,
        env,
        bridge: RLEnvEncoderBridge,
        policy_network: LatentPolicyNetwork,
        *,
        device: str = "mps",
        lr: float = 3e-4,
        gamma: float = 0.8,
        gae_lambda: float = 0.95,
        max_grad_norm: float = 0.5,
        ppo_clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        policy_updates: int = 4,
        n_vx_bins: int = 11,
        n_vy_bins: int = 5,
        vx_low: float = -3.0,
        vx_high: float = 3.0,
        vy_low: float = -0.2,
        vy_high: float = 0.75,
        seed: int | None = 42,
    ):
        self.env = env
        self.bridge = bridge
        self.policy_network = policy_network.to(device)
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.max_grad_norm = float(max_grad_norm)
        self.ppo_clip_eps = float(ppo_clip_eps)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.policy_updates = max(1, int(policy_updates))
        self.n_vx_bins = max(1, int(n_vx_bins))
        self.n_vy_bins = max(1, int(n_vy_bins))
        self.action_grid = _build_action_grid(
            vx_low=float(vx_low),
            vx_high=float(vx_high),
            n_vx_bins=self.n_vx_bins,
            vy_low=float(vy_low),
            vy_high=float(vy_high),
            n_vy_bins=self.n_vy_bins,
        )
        self.n_actions = int(self.action_grid.shape[0])
        self.rng = np.random.default_rng(seed if seed is not None else None)

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.reset_buffers()

    def reset_buffers(self):
        self.latents = []
        self.action_idx = []
        self.logp_old = []
        self.rewards = []
        self.values = []
        self.terminated = []

    def collect_trajectory(self, max_steps=1000, seed=None):
        self.reset_buffers()
        obs, _ = self.env.reset(seed=seed)
        self.bridge.reset(self.env)

        total_reward = 0.0
        steps = 0
        termination_reason = None
        end_distance = None
        sum_min_clearance = 0.0
        n_clearance = 0
        low_clearance_steps = 0
        last_next_obs = None
        term_sums: dict[str, float] = {}

        while steps < max_steps:
            z_t = self.bridge.encode_observation_tensor(obs).to(self.device)
            self.policy_network.eval()
            with torch.inference_mode():
                v, logits = self.policy_network(z_t)
                logp_all = torch.log_softmax(logits, dim=-1)
                probs = torch.exp(logp_all).squeeze(0).float().cpu().numpy()

            a_idx = int(self.rng.choice(self.n_actions, p=probs))
            action = self.action_grid[a_idx].astype(np.float32, copy=True)
            logp = float(logp_all.squeeze(0)[a_idx].detach().cpu().item())

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            last_next_obs = next_obs
            done = terminated or truncated
            if done:
                termination_reason = info.get("termination_reason")
                end_distance = float(info.get("distance_to_goal", np.nan))
            mc = info.get("min_clearance")
            if mc is not None and np.isfinite(mc):
                sum_min_clearance += float(mc)
                n_clearance += 1
                if float(mc) < 0.35:
                    low_clearance_steps += 1

            terms = getattr(self.env, "_last_reward_terms", None)
            if isinstance(terms, dict):
                for tk, tv in terms.items():
                    if isinstance(tv, (bool, np.bool_)):
                        continue
                    try:
                        fv = float(tv)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(fv):
                        term_sums[tk] = term_sums.get(tk, 0.0) + fv

            self.latents.append(z_t.squeeze(0).detach().cpu().numpy().astype(np.float32).copy())
            self.action_idx.append(a_idx)
            self.logp_old.append(logp)
            self.rewards.append(float(reward))
            self.values.append(float(v.item()))
            self.terminated.append(bool(terminated))

            total_reward += float(reward)
            steps += 1
            if done:
                break
            obs = next_obs

        if steps > 0 and bool(self.terminated[-1]) is False and last_next_obs is not None:
            z_next = self.bridge.encode_observation_tensor(last_next_obs).to(self.device)
            self.policy_network.eval()
            with torch.inference_mode():
                v_boot, _ = self.policy_network(z_next)
            boot_v = float(v_boot.item())
        else:
            boot_v = 0.0
        self.values.append(boot_v)

        denom = max(int(steps), 1)
        reward_term_means = {k: float(term_sums[k]) / float(denom) for k in sorted(term_sums.keys())}
        ep_stats = {
            "mean_min_clearance": float(sum_min_clearance / max(n_clearance, 1)),
            "low_clearance_frac": float(low_clearance_steps / max(n_clearance, 1)),
            "reward_term_means": reward_term_means,
        }
        return total_reward, steps, termination_reason, end_distance, ep_stats

    def compute_gae(self) -> tuple[np.ndarray, np.ndarray]:
        rewards = np.array(self.rewards, dtype=np.float64)
        values = np.array(self.values, dtype=np.float64)
        terminated = np.array(self.terminated, dtype=np.float64)
        if rewards.size == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        if int(values.size) != int(rewards.size + 1):
            raise RuntimeError(f"PPO buffers: len(values) must be len(rewards)+1, got {values.size} vs {rewards.size}")

        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            if terminated[t]:
                last_gae = 0.0
            nonterminal = 1.0 - terminated[t]
            delta = rewards[t] + self.gamma * values[t + 1] * nonterminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values[:-1]
        return advantages.astype(np.float32), returns.astype(np.float32)

    def update(self, advantages: np.ndarray, returns: np.ndarray, entropy_coef: float | None = None):
        z_batch = torch.from_numpy(np.stack(self.latents, axis=0).astype(np.float32)).to(self.device)
        a_batch = torch.from_numpy(np.asarray(self.action_idx, dtype=np.int64)).to(self.device)
        logp_old = torch.from_numpy(np.asarray(self.logp_old, dtype=np.float32)).to(self.device)

        adv = advantages.copy()
        if adv.size > 1:
            adv = (adv - float(adv.mean())) / (float(adv.std()) + 1e-6)
        ret = returns.copy()

        adv_t = torch.from_numpy(adv.astype(np.float32)).to(self.device)
        ret_t = torch.from_numpy(ret.astype(np.float32)).to(self.device)

        self.policy_network.train()
        last = {"policy_loss": None, "value_loss": None, "entropy": None}
        ent_coef = float(self.entropy_coef if entropy_coef is None else entropy_coef)

        for _ in range(self.policy_updates):
            v_pred, logits = self.policy_network(z_batch)
            logp_all = torch.log_softmax(logits, dim=-1)
            logp = logp_all.gather(1, a_batch.view(-1, 1)).squeeze(1)
            ratio = torch.exp(logp - logp_old)

            unclipped = ratio * adv_t
            clipped = torch.clamp(ratio, 1.0 - self.ppo_clip_eps, 1.0 + self.ppo_clip_eps) * adv_t
            policy_loss = -torch.min(unclipped, clipped).mean()

            value_loss = F.smooth_l1_loss(v_pred, ret_t)
            entropy = -(torch.exp(logp_all) * logp_all).sum(dim=-1).mean()

            loss = policy_loss + self.value_coef * value_loss - ent_coef * entropy
            if torch.isnan(loss):
                return last

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
            self.optimizer.step()

            last = {
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.item()),
            }

        self.reset_buffers()
        return last

    @property
    def policy_device(self) -> torch.device:
        return next(self.policy_network.parameters()).device

    def policy_state_dict(self) -> dict[str, torch.Tensor]:
        return self.policy_network.state_dict()

    def save_policy(self, path: str) -> None:
        torch.save(self.policy_state_dict(), path)
