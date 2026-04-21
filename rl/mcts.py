# rl/mcts.py — корневой MCTS в латенте по дискретной сетке (vx, vy)

from dataclasses import dataclass
import typing

import numpy as np
import torch
from rl.envencoder_bridge import RLEnvEncoderBridge


def discretize_vx_bins(vx_low: float, vx_high: float, n_actions: int) -> np.ndarray:
    """1D-центры vx (обратная совместимость; при n_vy_bins=1 предпочтительнее build_discrete_action_grid)."""
    if n_actions < 2:
        raise ValueError("n_actions >= 2")
    return np.linspace(vx_low, vx_high, n_actions, dtype=np.float32)


def build_discrete_action_grid(
    vx_low: float,
    vx_high: float,
    n_vx_bins: int,
    vy_low: float,
    vy_high: float,
    n_vy_bins: int,
) -> np.ndarray:
    """
    Сетка (vx, vy) в том же порядке, что и DQN: ``[(vx, vy) for vy in vy_centers for vx in vx_centers]``.
    Индекс a_idx соответствует flatten: vy_major, vx_minor (как train_dqn._build_action_grid).
    """
    if n_vx_bins < 1 or n_vy_bins < 1:
        raise ValueError("n_vx_bins >= 1, n_vy_bins >= 1")
    vx_centers = np.linspace(float(vx_low), float(vx_high), int(n_vx_bins), dtype=np.float32)
    vy_centers = np.linspace(float(vy_low), float(vy_high), int(n_vy_bins), dtype=np.float32)
    return np.array([(vx, vy) for vy in vy_centers for vx in vx_centers], dtype=np.float32)


@dataclass
class MCTSResult:
    """Выбранное действие [vx, vy] и статистика по всем бинам сетки."""

    action: np.ndarray  # shape (2,), float32
    visit_counts: np.ndarray
    action_centers: np.ndarray  # shape (n_actions, 2)

    @property
    def action_vx(self) -> float:
        return float(self.action[0])

    @property
    def action_vy(self) -> float:
        return float(self.action[1])


def root_mcts_search(
    env,
    bridge: "RLEnvEncoderBridge",
    value_net: torch.nn.Module,
    env_snapshot: dict,
    odom_snapshot: dict,
    *,
    vx_low: float,
    vx_high: float,
    n_vx_bins: int,
    vy_low: float,
    vy_high: float,
    n_vy_bins: int,
    n_simulations: int,
    gamma: float,
    c_puct: float,
    device: torch.device,
    prior_logits: typing.Optional[torch.Tensor] = None,
    restore_fn: typing.Callable[[dict, dict], None],
    root_dirichlet_alpha: float = 0.0,
    root_dirichlet_eps: float = 0.0,
    visit_temperature: float = 0.0,
    rng: typing.Optional[np.random.Generator] = None,
) -> MCTSResult:
    """
    prior_logits: [1, n_actions] или None --> равномерный prior.
    n_actions = n_vx_bins * n_vy_bins; порядок бинов — как ``build_discrete_action_grid`` / DQN.
    restore_fn(s_env, s_odom): восстановить среду и bridge.odom перед каждой симуляцией.
    """
    centers = build_discrete_action_grid(
        vx_low, vx_high, n_vx_bins, vy_low, vy_high, n_vy_bins
    )
    n_actions = int(centers.shape[0])
    if prior_logits is not None:
        n_log = int(prior_logits.shape[-1])
        if n_log != n_actions:
            raise ValueError(
                f"prior_logits last dim {n_log} != n_actions {n_actions} "
                f"(vxxvy = {n_vx_bins}x{n_vy_bins})"
            )
        with torch.inference_mode():
            p = torch.softmax(prior_logits.squeeze(0), dim=0).float().cpu().numpy().astype(np.float64)
        p = np.clip(p, 1e-8, 1.0)
        p = p / p.sum()
    else:
        p = np.ones(n_actions, dtype=np.float64) / n_actions
    if root_dirichlet_eps > 0.0 and root_dirichlet_alpha > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        noise = rng.dirichlet(np.full(n_actions, float(root_dirichlet_alpha), dtype=np.float64))
        eps = float(np.clip(root_dirichlet_eps, 0.0, 1.0))
        p = (1.0 - eps) * p + eps * noise
        p = np.clip(p, 1e-8, 1.0)
        p = p / p.sum()

    n_visits = np.zeros(n_actions, dtype=np.int64)
    value_sum = np.zeros(n_actions, dtype=np.float64)

    for _ in range(n_simulations):
        parent_visits = int(n_visits.sum())
        q = np.divide(
            value_sum,
            np.maximum(n_visits, 1).astype(np.float64),
            out=np.zeros_like(value_sum, dtype=np.float64),
            where=n_visits > 0,
        )
        ucb_scores = q + c_puct * p * np.sqrt(max(parent_visits, 1)) / (1.0 + n_visits.astype(np.float64))

        a_idx = int(np.argmax(ucb_scores))

        restore_fn(env_snapshot, odom_snapshot)

        cmd = centers[a_idx].astype(np.float32, copy=True)
        next_obs, reward, terminated, truncated, _ = env.step(cmd)
        done = terminated or truncated
        if done:
            g = float(reward)
        else:
            z_next_t = bridge.encode_observation_tensor(next_obs).to(device)
            with torch.inference_mode():
                out = value_net(z_next_t)
                if isinstance(out, tuple):
                    v = out[0]
                else:
                    v = out.squeeze(-1)
            g = float(reward) + gamma * float(v.item())

        n_visits[a_idx] += 1
        value_sum[a_idx] += g

    if visit_temperature > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        probs = n_visits.astype(np.float64)
        if probs.sum() <= 0:
            probs = np.ones_like(probs, dtype=np.float64)
        probs = np.power(probs, 1.0 / float(visit_temperature))
        probs_sum = float(probs.sum())
        if probs_sum <= 0.0 or not np.isfinite(probs_sum):
            probs = np.ones_like(probs, dtype=np.float64) / float(len(probs))
        else:
            probs = probs / probs_sum
        best = int(rng.choice(len(probs), p=probs))
    else:
        best = int(np.argmax(n_visits))
    restore_fn(env_snapshot, odom_snapshot)
    return MCTSResult(
        action=centers[best].astype(np.float32, copy=True),
        visit_counts=n_visits.astype(np.float32),
        action_centers=centers,
    )
