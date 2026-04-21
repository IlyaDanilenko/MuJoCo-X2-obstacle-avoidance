# scenaries/train_mcts.py - обучение MCTS+латент (V и prior по сетке vxxvy)

import os
from collections import deque
import typing

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rl.environment import DroneObstacleAvoidanceEnv
from rl.envencoder_bridge import RLEnvEncoderBridge
from rl.mcts import root_mcts_search
from rl.ppo import LatentPolicyNetwork
from utils.reward_config import (
    apply_curriculum_stage_full,
    apply_env_reward_dict,
    build_curriculum_stages,
    dqn_gamma_from_reward_cfg,
    load_reward_config,
    merge_map_train_args_from_reward_yaml,
    merge_mcts_train_args_from_reward_yaml,
)
from utils.training_logging import (
    copy_reward_yaml_into_run,
    summarize_curriculum_stages,
    write_train_run_config_yaml,
)


def _apply_curriculum_stage(env: DroneObstacleAvoidanceEnv, stage: dict) -> None:
    apply_curriculum_stage_full(env, stage)


def _same_torch_device(a: torch.device, b: torch.device) -> bool:
    if a.type != b.type:
        return False
    ai = 0 if a.index is None else a.index
    bi = 0 if b.index is None else b.index
    return ai == bi


class MCTSLatentTrainer:
    """
    Сбор траекторий: латент z = EnvEncoder(obs), действие (vx, vy) из корневого MCTS
    (PUCT + prior по дискретной сетке из сети).
    Обучение: MSE(ценность V(латент), returns) + CE(prior по бинам сетки, визиты MCTS).
    """

    def __init__(
        self,
        env,
        bridge: RLEnvEncoderBridge,
        policy_network: LatentPolicyNetwork,
        device="mps",
        lr: float = 3e-4,
        gamma=0.8,
        gae_lambda=0.9,
        max_grad_norm=0.5,
        vx_low: float = -3.0,
        vx_high: float = 3.0,
        n_vx_bins: int = 11,
        vy_low: float = 0.01,
        vy_high: float = 0.75,
        n_vy_bins: int = 5,
        n_mcts_simulations: int = 12,
        mcts_c_puct: float = 1.5,
        value_coef: float = 1.0,
        prior_coef: float = 0.5,
        policy_updates: int = 2,
        root_dirichlet_alpha: float = 0.3,
        root_dirichlet_eps: float = 0.25,
        visit_temperature: float = 1.0,
        prior_label_smoothing: float = 0.05,
        seed: int | None = 42,
    ):
        self.env = env
        self.bridge = bridge
        self.policy_network = policy_network.to(device)
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.vx_low = float(vx_low)
        self.vx_high = float(vx_high)
        self.n_vx_bins = int(n_vx_bins)
        self.vy_low = float(vy_low)
        self.vy_high = float(vy_high)
        self.n_vy_bins = int(n_vy_bins)
        self.n_actions = int(n_vx_bins) * int(n_vy_bins)
        self.n_mcts_simulations = int(n_mcts_simulations)
        self.mcts_c_puct = float(mcts_c_puct)
        self.value_coef = float(value_coef)
        self.prior_coef = float(prior_coef)
        self.policy_updates = max(1, int(policy_updates))
        self.root_dirichlet_alpha = float(max(0.0, root_dirichlet_alpha))
        self.root_dirichlet_eps = float(np.clip(root_dirichlet_eps, 0.0, 1.0))
        self.visit_temperature = float(max(0.0, visit_temperature))
        self.prior_label_smoothing = float(np.clip(prior_label_smoothing, 0.0, 1.0))
        self.rng = np.random.default_rng(seed if seed is not None else None)

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=float(lr))
        self.reset_buffers()

    def set_vy_range(self, vy_low: float, vy_high: float) -> None:
        self.vy_low = float(vy_low)
        self.vy_high = float(vy_high)

    def reset_buffers(self):
        self.latents = []
        self.actions = []
        self.visit_targets = []
        self.rewards = []
        self.values = []
        self.dones = []

    def _restore(self, env_snap, odom_snap):
        self.env.restore_from_branching_snapshot(env_snap)
        self.bridge.unpack_odom_state(odom_snap)

    def collect_trajectory(self, max_steps=1000, seed=None):
        # Сброс до сбора: защита от повторного append без прошлого reset; сразу освобождаем ссылки
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

        while steps < max_steps:
            z_t = self.bridge.encode_observation_tensor(obs).to(self.device)
            env_snap = self.env.snapshot_for_branching()
            odom_snap = self.bridge.pack_odom_state()

            self.policy_network.eval()
            with torch.inference_mode():
                v, logits = self.policy_network(z_t)

            mcts_out = root_mcts_search(
                self.env,
                self.bridge,
                self.policy_network,
                env_snap,
                odom_snap,
                vx_low=self.vx_low,
                vx_high=self.vx_high,
                n_vx_bins=self.n_vx_bins,
                vy_low=self.vy_low,
                vy_high=self.vy_high,
                n_vy_bins=self.n_vy_bins,
                n_simulations=self.n_mcts_simulations,
                gamma=self.gamma,
                c_puct=self.mcts_c_puct,
                device=self.device,
                prior_logits=logits,
                restore_fn=self._restore,
                root_dirichlet_alpha=self.root_dirichlet_alpha,
                root_dirichlet_eps=self.root_dirichlet_eps,
                visit_temperature=self.visit_temperature,
                rng=self.rng,
            )
            action = mcts_out.action.copy()
            visits = mcts_out.visit_counts
            vis_sum = float(visits.sum()) + 1e-8
            pi_target = visits / vis_sum
            if self.prior_label_smoothing > 0.0:
                u = np.full_like(pi_target, 1.0 / float(self.n_actions), dtype=np.float32)
                s = self.prior_label_smoothing
                pi_target = (1.0 - s) * pi_target + s * u

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            mc = info.get("min_clearance")
            if mc is not None and np.isfinite(mc):
                sum_min_clearance += float(mc)
                n_clearance += 1
                if float(mc) < 0.35:
                    low_clearance_steps += 1

            self.latents.append(
                z_t.squeeze(0).detach().cpu().numpy().astype(np.float32).copy()
            )
            self.actions.append(action.copy())
            self.visit_targets.append(pi_target.astype(np.float32))
            self.rewards.append(reward)
            self.values.append(float(v.item()))
            self.dones.append(done)

            total_reward += reward
            steps += 1
            if done:
                termination_reason = info.get("termination_reason")
                end_distance = float(info.get("distance_to_goal", np.nan))
                break
            obs = next_obs

        ep_stats = {
            "mean_min_clearance": float(sum_min_clearance / max(n_clearance, 1)),
            "low_clearance_frac": float(low_clearance_steps / max(n_clearance, 1)),
        }
        ep_info: dict[str, typing.Any] = {
            "termination_reason": termination_reason,
            "end_distance": end_distance,
            **ep_stats,
        }
        return total_reward, steps, ep_info

    def compute_gae(self, next_value=0.0):
        rewards = np.array(self.rewards, dtype=np.float64)
        values = np.array(self.values + [next_value], dtype=np.float64)
        dones = np.array(self.dones, dtype=np.float64)

        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                last_gae = 0.0
            delta = rewards[t] + self.gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values[:-1]
        return advantages.tolist(), returns.tolist()

    def update(self, advantages, returns):
        # Стабилизируем масштаб value-target между эпизодами разной длины/награды.
        returns_arr = np.asarray(returns, dtype=np.float32)
        if returns_arr.size > 1:
            r_mean = float(returns_arr.mean())
            r_std = float(returns_arr.std())
            returns_arr = (returns_arr - r_mean) / (r_std + 1e-6)
        returns_t = torch.from_numpy(returns_arr).to(self.device)
        z_batch = torch.from_numpy(np.stack(self.latents, axis=0).astype(np.float32)).to(self.device)
        pi_target = torch.from_numpy(np.stack(self.visit_targets, axis=0).astype(np.float32)).to(self.device)

        self.latents.clear()
        self.visit_targets.clear()

        self.policy_network.train()
        last_v_loss = None
        last_p_loss = None

        for _ in range(self.policy_updates):
            v_pred, logits = self.policy_network(z_batch)
            # Huber менее чувствителен к редким выбросам target, чем MSE.
            value_loss = nn.SmoothL1Loss()(v_pred, returns_t)
            logp = torch.log_softmax(logits, dim=-1)
            prior_loss = -(pi_target * logp).sum(dim=-1).mean()

            loss = self.value_coef * value_loss + self.prior_coef * prior_loss

            if torch.isnan(loss):
                return {"value_loss": None, "prior_loss": None}

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
            self.optimizer.step()

            last_v_loss = float(value_loss.item())
            last_p_loss = float(prior_loss.item())

        # Сразу отпускаем траекторию (до конца итерации в train()); меньше пик RAM и давление на GC
        self.rewards.clear()
        self.actions.clear()
        self.values.clear()
        self.dones.clear()

        return {"value_loss": last_v_loss, "prior_loss": last_p_loss}


def train(
    num_episodes=1000,
    max_steps_per_episode=1000,
    save_interval=100,
    model_dir="models/mcts-latent",
    log_dir=None,
    device="mps",
    num_obstacles=5,
    encoder_path: str | None = None,
    fastdepth_weights=None,
    n_mcts_simulations=12,
    n_vx_bins: int = 11,
    n_vy_bins: int = 5,
    vy_min: float = 0.01,
    vy_max: float = 0.75,
    mcts_c_puct=1.5,
    policy_updates: int = 2,
    root_dirichlet_alpha: float = 0.3,
    root_dirichlet_eps: float = 0.25,
    visit_temperature: float = 1.0,
    prior_label_smoothing: float = 0.05,
    gamma: typing.Optional[float] = None,
    lr: float = 3e-4,
    max_grad_norm: float = 0.5,
    value_coef: float = 1.0,
    prior_coef: float = 0.5,
    gae_lambda: float = 0.9,
    seed: int | None = 42,
    path_length: float = 5.0,
    map_type: str = "random",
    reward_config_path: str | None = None,
    use_curriculum: bool = True,
    curriculum_success_threshold: typing.Optional[float] = None,
    curriculum_window: typing.Optional[int] = None,
    curriculum_min_episodes_per_stage: typing.Optional[int] = None,
    curriculum_use_fallback: bool = True,
    curriculum_fallback_max_mean_dist: float = 0.55,
    curriculum_fallback_max_collision_rate: float = 0.38,
    early_stop_patience: int = 20,
    early_stop_min_episodes: int = 30,
):
    if encoder_path is None or not os.path.isfile(encoder_path):
        raise FileNotFoundError(
            "Укажите путь к весам EnvEncoder (--encoder-path), файл не найден: "
            f"{encoder_path!r}"
        )

    if device == "mps" and not torch.backends.mps.is_available():
        print("⚠ MPS недоступен, используем CPU")
        device = "cpu"
    cst_user = curriculum_success_threshold
    cw_user = curriculum_window
    cmin_user = curriculum_min_episodes_per_stage
    curriculum_success_threshold = 0.8 if cst_user is None else float(cst_user)
    curriculum_window = 40 if cw_user is None else int(cw_user)
    curriculum_min_episodes_per_stage = 30 if cmin_user is None else int(cmin_user)

    if seed is not None:
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    model_dir = os.path.abspath(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    existing = []
    if os.path.isdir(model_dir):
        for name in os.listdir(model_dir):
            path = os.path.join(model_dir, name)
            if os.path.isdir(path) and (
                name == "run" or (name.startswith("run") and len(name) > 3 and name[3:].isdigit())
            ):
                existing.append(name)
    run_numbers = [0 if n == "run" else int(n[3:]) for n in existing]
    next_num = max(run_numbers) + 1 if run_numbers else 0
    run_name = "run" if next_num == 0 else f"run{next_num}"
    run_dir = os.path.join(model_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"✓ Директория для моделей: {model_dir}")
    print(f"✓ Текущий запуск: {run_dir}")

    reward_copy_path, reward_yaml_resolved = copy_reward_yaml_into_run(run_dir, reward_config_path)
    reward_cfg = load_reward_config(reward_yaml_resolved)
    print(f"✓ Копия наград в run: {reward_copy_path}")

    merged = merge_mcts_train_args_from_reward_yaml(
        reward_cfg,
        {
            "gamma": gamma,
            "n_mcts_simulations": n_mcts_simulations,
            "n_vx_bins": n_vx_bins,
            "n_vy_bins": n_vy_bins,
            "vy_min": vy_min,
            "vy_max": vy_max,
            "mcts_c_puct": mcts_c_puct,
            "policy_updates": policy_updates,
            "root_dirichlet_alpha": root_dirichlet_alpha,
            "root_dirichlet_eps": root_dirichlet_eps,
            "visit_temperature": visit_temperature,
            "prior_label_smoothing": prior_label_smoothing,
            "lr": lr,
            "max_grad_norm": max_grad_norm,
            "value_coef": value_coef,
            "prior_coef": prior_coef,
            "gae_lambda": gae_lambda,
            "curriculum_success_threshold": curriculum_success_threshold,
            "curriculum_window": curriculum_window,
            "curriculum_min_episodes_per_stage": curriculum_min_episodes_per_stage,
            "curriculum_use_fallback": curriculum_use_fallback,
            "curriculum_fallback_max_mean_dist": curriculum_fallback_max_mean_dist,
            "curriculum_fallback_max_collision_rate": curriculum_fallback_max_collision_rate,
            "early_stop_patience": early_stop_patience,
            "early_stop_min_episodes": early_stop_min_episodes,
        },
    )
    gamma = merged["gamma"]
    n_mcts_simulations = merged["n_mcts_simulations"]
    n_vx_bins = merged["n_vx_bins"]
    n_vy_bins = merged["n_vy_bins"]
    vy_min = merged["vy_min"]
    vy_max = merged["vy_max"]
    mcts_c_puct = merged["mcts_c_puct"]
    policy_updates = merged["policy_updates"]
    root_dirichlet_alpha = merged["root_dirichlet_alpha"]
    root_dirichlet_eps = merged["root_dirichlet_eps"]
    visit_temperature = merged["visit_temperature"]
    prior_label_smoothing = merged["prior_label_smoothing"]
    lr = merged["lr"]
    max_grad_norm = merged["max_grad_norm"]
    value_coef = merged["value_coef"]
    prior_coef = merged["prior_coef"]
    gae_lambda = merged["gae_lambda"]
    curriculum_success_threshold = merged["curriculum_success_threshold"]
    curriculum_window = merged["curriculum_window"]
    curriculum_min_episodes_per_stage = merged["curriculum_min_episodes_per_stage"]
    curriculum_use_fallback = merged["curriculum_use_fallback"]
    curriculum_fallback_max_mean_dist = merged["curriculum_fallback_max_mean_dist"]
    curriculum_fallback_max_collision_rate = merged["curriculum_fallback_max_collision_rate"]
    early_stop_patience = merged["early_stop_patience"]
    early_stop_min_episodes = merged["early_stop_min_episodes"]
    map_merged = merge_map_train_args_from_reward_yaml(
        reward_cfg,
        {
            "map_type": map_type,
            "path_length": path_length,
            "num_obstacles": num_obstacles,
            "vy_min": vy_min,
            "vy_max": vy_max,
            "n_vx_bins": n_vx_bins,
            "n_vy_bins": n_vy_bins,
        },
    )
    map_type = map_merged["map_type"]
    path_length = map_merged["path_length"]
    num_obstacles = map_merged["num_obstacles"]
    vy_min = map_merged["vy_min"]
    vy_max = map_merged["vy_max"]
    n_vx_bins = map_merged["n_vx_bins"]
    n_vy_bins = map_merged["n_vy_bins"]

    if gamma is None:
        g_yaml = dqn_gamma_from_reward_cfg(reward_cfg)
        if g_yaml is not None:
            gamma = g_yaml
    if gamma is None:
        gamma = 0.8
    gamma = float(gamma)

    if log_dir is None:
        log_dir = os.path.join(run_dir, "tensorboard")
    log_dir = os.path.abspath(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"✓ TensorBoard: {log_dir}")
    print(
        f"MCTS: γ={gamma}, sims={n_mcts_simulations}, Adam/эпизод={policy_updates}, "
        f"actions={n_vx_bins}x{n_vy_bins}, reward_config={reward_yaml_resolved}"
    )
    print(
        "Exploration: "
        f"dir(alpha={root_dirichlet_alpha}, eps={root_dirichlet_eps}), "
        f"temp={visit_temperature}, smoothing={prior_label_smoothing}"
    )

    env = DroneObstacleAvoidanceEnv(
        render_mode=None,
        max_velocity=3.0,
        forward_velocity=0.25,
        max_episode_steps=max_steps_per_episode,
        num_obstacles=num_obstacles,
        obstacle_path_length=float(path_length),
        show_training_target_pillar=False,
        map_type=map_type,
    )
    apply_env_reward_dict(env, reward_cfg.get("env", {}))
    if use_curriculum:
        raw_curr = reward_cfg.get("curriculum") or []
        curriculum = (
            build_curriculum_stages(raw_curr, path_length, num_obstacles)
            if raw_curr
            else None
        )
    else:
        curriculum = None
    stage_idx = 0
    stage_start_episode = 1
    if curriculum is not None:
        _apply_curriculum_stage(env, curriculum[stage_idx])
        print(f"✓ Curriculum: stage 1/{len(curriculum)} = {curriculum[stage_idx]['name']}")

    bridge = RLEnvEncoderBridge(
        encoder_path=encoder_path,
        device=device,
        fastdepth_weights=fastdepth_weights,
    )
    n_actions = int(n_vx_bins) * int(n_vy_bins)
    policy = LatentPolicyNetwork(latent_dim=bridge.latent_dim, n_actions=n_actions)
    trainer = MCTSLatentTrainer(
        env=env,
        bridge=bridge,
        policy_network=policy,
        device=device,
        lr=float(lr),
        gamma=gamma,
        gae_lambda=float(gae_lambda),
        max_grad_norm=float(max_grad_norm),
        value_coef=float(value_coef),
        prior_coef=float(prior_coef),
        vx_low=float(env.action_space.low[0]),
        vx_high=float(env.action_space.high[0]),
        n_vx_bins=n_vx_bins,
        vy_low=float(vy_min),
        vy_high=float(vy_max),
        n_vy_bins=n_vy_bins,
        n_mcts_simulations=n_mcts_simulations,
        mcts_c_puct=mcts_c_puct,
        policy_updates=policy_updates,
        root_dirichlet_alpha=root_dirichlet_alpha,
        root_dirichlet_eps=root_dirichlet_eps,
        visit_temperature=visit_temperature,
        prior_label_smoothing=prior_label_smoothing,
        seed=seed,
    )

    run_cfg_path = write_train_run_config_yaml(
        run_dir=run_dir,
        algorithm="mcts",
        env=env,
        bridge=bridge,
        encoder_path=encoder_path,
        device=device,
        seed=seed,
        policy_inference={
            "network": "LatentPolicyNetwork",
            "latent_dim": int(bridge.latent_dim),
            "n_actions": int(n_actions),
            "discretization": {
                "n_vx_bins": int(n_vx_bins),
                "n_vy_bins": int(n_vy_bins),
                "vy_min": float(vy_min),
                "vy_max": float(vy_max),
                "vx_low": float(env.action_space.low[0]),
                "vx_high": float(env.action_space.high[0]),
            },
            "mcts": {
                "n_mcts_simulations": int(trainer.n_mcts_simulations),
                "c_puct": float(trainer.mcts_c_puct),
                "gamma": float(trainer.gamma),
                "root_dirichlet_alpha": float(trainer.root_dirichlet_alpha),
                "root_dirichlet_eps": float(trainer.root_dirichlet_eps),
                "visit_temperature": float(trainer.visit_temperature),
                "prior_label_smoothing": float(trainer.prior_label_smoothing),
            },
        },
        training_hyperparams={
            "lr": float(trainer.optimizer.param_groups[0]["lr"]),
            "policy_updates": int(trainer.policy_updates),
            "gamma": float(trainer.gamma),
            "gae_lambda": float(trainer.gae_lambda),
            "max_grad_norm": float(trainer.max_grad_norm),
            "value_coef": float(trainer.value_coef),
            "prior_coef": float(trainer.prior_coef),
            "early_stop_patience": int(early_stop_patience),
            "early_stop_min_episodes": int(early_stop_min_episodes),
            "curriculum_success_threshold": float(curriculum_success_threshold),
            "curriculum_window": int(curriculum_window),
            "curriculum_min_episodes_per_stage": int(curriculum_min_episodes_per_stage),
            "curriculum_use_fallback": bool(curriculum_use_fallback),
            "curriculum_fallback_max_mean_dist": float(curriculum_fallback_max_mean_dist),
            "curriculum_fallback_max_collision_rate": float(curriculum_fallback_max_collision_rate),
        },
        curriculum_stages=summarize_curriculum_stages(curriculum),
        use_curriculum=use_curriculum,
    )
    print(f"✓ Конфиг run: {run_cfg_path}")

    enc_dev = bridge.device_t
    pol_dev = next(trainer.policy_network.parameters()).device
    print("Модели и устройства:")
    print(f"  FastDepth / EnvEncoder --> {enc_dev}")
    print(f"  Критик V(латент) + prior(vxxvy) --> {pol_dev}")
    if not _same_torch_device(enc_dev, pol_dev):
        print("⚠ Латент и политика на разных устройствах (редкий случай).")

    episode_rewards = deque(maxlen=100)
    success_window = deque(maxlen=max(5, int(curriculum_window)))
    goal_window = deque(maxlen=max(5, int(curriculum_window)))
    collision_window = deque(maxlen=max(5, int(curriculum_window)))
    timeout_window = deque(maxlen=max(5, int(curriculum_window)))
    out_window = deque(maxlen=max(5, int(curriculum_window)))
    end_dist_window = deque(maxlen=max(5, int(curriculum_window)))
    best_avg_reward = -float("inf")
    best_path = os.path.join(run_dir, "latent_policy_best.pth")
    no_improve = 0

    pbar = tqdm(
        range(num_episodes),
        desc="Обучение (MCTS+латент)",
        unit="эпизод",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for episode in pbar:
        if curriculum is not None:
            st = curriculum[stage_idx]
            trainer.set_vy_range(
                float(st.get("vy_min", vy_min)),
                float(st.get("vy_max", vy_max)),
            )

        total_reward, steps, ep_info = trainer.collect_trajectory(max_steps_per_episode, seed=None)
        advantages, returns = trainer.compute_gae()
        metrics = trainer.update(advantages, returns)

        termination_reason = ep_info.get("termination_reason")
        end_distance = ep_info.get("end_distance")
        episode_rewards.append(total_reward)

        ep_num = episode + 1
        if termination_reason == "goal_reached":
            succ_p = os.path.join(run_dir, f"latent_policy_success_ep{ep_num}.pth")
            torch.save(policy.state_dict(), succ_p)
            tqdm.write(f"✓ Успех (goal): {succ_p}")

        success = 1.0 if termination_reason == "goal_reached" else 0.0
        collision = 1.0 if termination_reason == "collision" else 0.0
        timeout = 1.0 if termination_reason == "timeout" else 0.0
        out_fail = 1.0 if termination_reason in {"out_of_bounds", "out_of_corridor", "fallen"} else 0.0
        success_window.append(success)
        goal_window.append(success)
        collision_window.append(collision)
        timeout_window.append(timeout)
        out_window.append(out_fail)
        end_distance_f = end_distance
        if end_distance_f is not None and np.isfinite(end_distance_f):
            end_dist_window.append(float(end_distance_f))

        avg_reward = float(np.mean(episode_rewards))
        success_rate = float(np.mean(success_window)) if success_window else 0.0
        goal_rate = float(np.mean(goal_window)) if goal_window else 0.0
        collision_rate = float(np.mean(collision_window)) if collision_window else 0.0
        timeout_rate = float(np.mean(timeout_window)) if timeout_window else 0.0
        out_rate = float(np.mean(out_window)) if out_window else 0.0
        mean_end_dist = float(np.mean(end_dist_window)) if end_dist_window else float("nan")

        writer.add_scalar("episode/reward", total_reward, ep_num)
        writer.add_scalar("episode/length", steps, ep_num)
        writer.add_scalar("episode/avg_reward", avg_reward, ep_num)
        writer.add_scalar("episode/success", success, ep_num)
        writer.add_scalar("episode/success_rate_window", success_rate, ep_num)
        writer.add_scalar("episode/goal_rate_window", goal_rate, ep_num)
        writer.add_scalar("episode/collision_rate_window", collision_rate, ep_num)
        writer.add_scalar("episode/timeout_rate_window", timeout_rate, ep_num)
        writer.add_scalar("episode/out_rate_window", out_rate, ep_num)
        if np.isfinite(mean_end_dist):
            writer.add_scalar("episode/mean_dist_to_goal_end", mean_end_dist, ep_num)
        writer.add_scalar("episode/mean_min_clearance", float(ep_info["mean_min_clearance"]), ep_num)
        writer.add_scalar("episode/low_clearance_frac", float(ep_info["low_clearance_frac"]), ep_num)
        writer.add_scalar("curriculum/stage_idx", float(stage_idx), ep_num)
        if metrics.get("value_loss") is not None:
            writer.add_scalar("loss/value", metrics["value_loss"], ep_num)
        if metrics.get("prior_loss") is not None:
            writer.add_scalar("loss/prior", metrics["prior_loss"], ep_num)

        if curriculum is not None:
            st0 = curriculum[stage_idx]
            writer.add_scalar("curriculum/path_length", float(env.obstacle_path_length), ep_num)
            fb_coll0 = float(
                st0.get("fallback_max_collision_rate", curriculum_fallback_max_collision_rate)
            )
            fb_dist0 = float(st0.get("fallback_max_mean_dist", curriculum_fallback_max_mean_dist))
            frd0 = st0.get("fallback_relaxed_mean_dist")
            fro0 = float(st0.get("fallback_max_out_rate", 0.4))
            writer.add_scalar("curriculum/threshold_success_rate", float(curriculum_success_threshold), ep_num)
            writer.add_scalar(
                "curriculum/gap_success_vs_threshold",
                success_rate - float(curriculum_success_threshold),
                ep_num,
            )
            writer.add_scalar("curriculum/gap_collision_headroom", fb_coll0 - collision_rate, ep_num)
            writer.add_scalar("curriculum/gap_out_headroom", fro0 - out_rate, ep_num)
            if np.isfinite(mean_end_dist):
                writer.add_scalar(
                    "curriculum/gap_strict_dist_headroom",
                    fb_dist0 - mean_end_dist,
                    ep_num,
                )
                if frd0 is not None:
                    writer.add_scalar(
                        "curriculum/gap_relaxed_dist_headroom",
                        float(frd0) - mean_end_dist,
                        ep_num,
                    )

        pbar.set_postfix(
            {
                "R": f"{total_reward:.2f}",
                "avgR": f"{avg_reward:.2f}",
                "succ": f"{success_rate:.2f}",
                "coll": f"{collision_rate:.2f}",
                "stage": f"{stage_idx + 1}",
                "steps": f"{steps}",
            }
        )

        if curriculum is not None and stage_idx < (len(curriculum) - 1):
            stage_episodes = ep_num - stage_start_episode + 1
            st_cur = curriculum[stage_idx]
            fb_dist = float(st_cur.get("fallback_max_mean_dist", curriculum_fallback_max_mean_dist))
            fb_coll = float(st_cur.get("fallback_max_collision_rate", curriculum_fallback_max_collision_rate))
            win_ok = len(success_window) >= max(5, int(curriculum_window))
            ep_ok = stage_episodes >= int(curriculum_min_episodes_per_stage)
            by_success = win_ok and ep_ok and success_rate >= float(curriculum_success_threshold)
            by_fallback = (
                curriculum_use_fallback
                and win_ok
                and ep_ok
                and np.isfinite(mean_end_dist)
                and float(mean_end_dist) < fb_dist
                and collision_rate <= fb_coll
            )
            frd = st_cur.get("fallback_relaxed_mean_dist")
            fro = float(st_cur.get("fallback_max_out_rate", 0.4))
            by_fallback_relaxed = (
                curriculum_use_fallback
                and frd is not None
                and win_ok
                and ep_ok
                and np.isfinite(mean_end_dist)
                and float(mean_end_dist) < float(frd)
                and collision_rate <= fb_coll
                and out_rate <= fro
            )
            if by_success or by_fallback or by_fallback_relaxed:
                stage_idx += 1
                _apply_curriculum_stage(env, curriculum[stage_idx])
                stage_start_episode = ep_num + 1
                success_window.clear()
                goal_window.clear()
                collision_window.clear()
                timeout_window.clear()
                out_window.clear()
                end_dist_window.clear()
                if by_success:
                    adv_tag = "success_rate"
                    adv_code = 1.0
                elif by_fallback:
                    adv_tag = "fallback_dist_collision"
                    adv_code = 2.0
                else:
                    adv_tag = "fallback_relaxed_out_dist"
                    adv_code = 3.0
                writer.add_scalar("curriculum/advance_reason", adv_code, ep_num)
                tqdm.write(
                    f"⇧ Curriculum stage {stage_idx + 1}/{len(curriculum)}: {curriculum[stage_idx]['name']} "
                    f"({adv_tag}; succ={success_rate:.2f}, mean_dist={mean_end_dist:.3f}, "
                    f"coll={collision_rate:.2f}, out={out_rate:.2f})"
                )

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            torch.save(policy.state_dict(), best_path)
            writer.add_scalar("best/avg_reward", best_avg_reward, ep_num)
            tqdm.write(f"★ Лучшая модель (avgR={best_avg_reward:.2f}): {best_path}")
            no_improve = 0
        else:
            no_improve += 1

        if (episode + 1) % save_interval == 0:
            pth = os.path.join(run_dir, f"latent_policy_ep{episode + 1}.pth")
            torch.save(policy.state_dict(), pth)
            tqdm.write(f"✓ Сохранено: {pth}")

        trainer.reset_buffers()
        if device == "mps" and (episode + 1) % 25 == 0 and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        if (
            ep_num >= int(early_stop_min_episodes)
            and no_improve >= int(early_stop_patience)
            and (curriculum is None or stage_idx >= (len(curriculum) - 1))
        ):
            tqdm.write(
                "⏹ Early stop: "
                f"{no_improve} эпизодов без улучшения avgR (best={best_avg_reward:.2f})"
            )
            break

    pbar.close()
    final = os.path.join(run_dir, "latent_policy_final.pth")
    torch.save(policy.state_dict(), final)
    writer.close()
    print(f"\n✓ Готово. Модель: {final}")
    env.close()
