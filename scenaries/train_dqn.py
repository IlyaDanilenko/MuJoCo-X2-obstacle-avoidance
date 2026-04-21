# scenaries/train_dqn.py - DQN в латенте (сетка vxxvy)

import os
from collections import deque
import typing

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rl.environment import DroneObstacleAvoidanceEnv
from rl.envencoder_bridge import RLEnvEncoderBridge
from rl.dqn import LatentQNetwork
from utils.reward_config import (
    apply_curriculum_stage_full,
    apply_env_reward_dict,
    build_curriculum_stages,
    dqn_epsilon_decay_steps_from_reward_cfg,
    dqn_gamma_from_reward_cfg,
    load_reward_config,
    merge_map_train_args_from_reward_yaml,
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
        return (
            self.z[idxs],
            self.a[idxs],
            self.r[idxs],
            self.z_next[idxs],
            self.done[idxs],
        )


def train(
    num_episodes=1000,
    max_steps_per_episode=1000,
    save_interval=100,
    model_dir="models/dqn-latent",
    log_dir=None,
    device="mps",
    num_obstacles=5,
    encoder_path: str | None = None,
    fastdepth_weights=None,
    n_vx_bins=11,
    n_vy_bins=5,
    vy_min: float = -0.2,
    vy_max: float = 0.75,
    gamma: typing.Optional[float] = None,
    lr: float = 1e-4,
    max_grad_norm: float = 10.0,
    seed: int | None = 42,
    path_length: float = 5.0,
    map_type: str = "random",
    early_stop_patience: int = 20,
    early_stop_min_episodes: int = 30,
    use_curriculum: bool = True,
    curriculum_success_threshold: typing.Optional[float] = None,
    curriculum_window: typing.Optional[int] = None,
    curriculum_min_episodes_per_stage: typing.Optional[int] = None,
    curriculum_use_fallback: bool = True,
    curriculum_fallback_max_mean_dist: float = 0.55,
    curriculum_fallback_max_collision_rate: float = 0.38,
    reward_config_path: str | None = None,
    batch_size: int = 128,
    buffer_capacity: int = 200_000,
    learning_starts: int = 10_000,
    train_frequency: int = 4,
    target_update_every: int = 1000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: typing.Optional[int] = None,
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
    rng = np.random.default_rng(seed)

    total_plan_steps = max(1, int(num_episodes) * int(max_steps_per_episode))

    model_dir = os.path.abspath(model_dir)
    os.makedirs(model_dir, exist_ok=True)
    existing = []
    for name in os.listdir(model_dir):
        path = os.path.join(model_dir, name)
        if os.path.isdir(path) and (name == "run" or (name.startswith("run") and name[3:].isdigit())):
            existing.append(name)
    run_numbers = [0 if n == "run" else int(n[3:]) for n in existing]
    next_num = max(run_numbers) + 1 if run_numbers else 0
    run_name = "run" if next_num == 0 else f"run{next_num}"
    run_dir = os.path.join(model_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    reward_copy_path, reward_yaml_resolved = copy_reward_yaml_into_run(run_dir, reward_config_path)
    reward_cfg = load_reward_config(reward_yaml_resolved)

    if gamma is None:
        g_yaml = dqn_gamma_from_reward_cfg(reward_cfg)
        if g_yaml is not None:
            gamma = g_yaml
    if gamma is None:
        gamma = 0.8
    gamma = float(gamma)

    if epsilon_decay_steps is None:
        from_yaml = dqn_epsilon_decay_steps_from_reward_cfg(reward_cfg)
        if from_yaml is not None:
            epsilon_decay_steps = from_yaml
    if epsilon_decay_steps is None:
        epsilon_decay_steps = total_plan_steps
    epsilon_decay_steps = max(1, int(epsilon_decay_steps))
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

    if log_dir is None:
        log_dir = os.path.join(run_dir, "tensorboard")
    log_dir = os.path.abspath(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    print(f"✓ DQN запуск: {run_dir}")
    print(f"✓ Копия наград: {reward_copy_path}")
    print(f"✓ TensorBoard: {log_dir}")
    print(
        f"DQN: γ={gamma}, lr={lr}, ε {epsilon_start}->{epsilon_end} за {epsilon_decay_steps} шагов, "
        f"batch={batch_size}, buffer={buffer_capacity}, actions={n_vx_bins}x{n_vy_bins}, "
        f"reward_config={reward_yaml_resolved}"
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
        print(f"✓ Curriculum включён: stage 1/{len(curriculum)} = {curriculum[stage_idx]['name']}")

    bridge = RLEnvEncoderBridge(
        encoder_path=encoder_path,
        device=device,
        fastdepth_weights=fastdepth_weights,
    )
    dev = torch.device(device)
    n_actions_total = int(n_vx_bins) * int(n_vy_bins)
    action_grid = _build_action_grid(
        vx_low=float(env.action_space.low[0]),
        vx_high=float(env.action_space.high[0]),
        n_vx_bins=int(n_vx_bins),
        vy_low=float(vy_min),
        vy_high=float(vy_max),
        n_vy_bins=int(n_vy_bins),
    )
    assert action_grid.shape[0] == n_actions_total

    run_cfg_path = write_train_run_config_yaml(
        run_dir=run_dir,
        algorithm="dqn",
        env=env,
        bridge=bridge,
        encoder_path=encoder_path,
        device=device,
        seed=seed,
        policy_inference={
            "network": "LatentQNetwork",
            "latent_dim": int(bridge.latent_dim),
            "n_actions": int(n_actions_total),
            "discretization": {
                "n_vx_bins": int(n_vx_bins),
                "n_vy_bins": int(n_vy_bins),
                "vy_min": float(vy_min),
                "vy_max": float(vy_max),
                "vx_low": float(env.action_space.low[0]),
                "vx_high": float(env.action_space.high[0]),
            },
        },
        training_hyperparams={
            "gamma": float(gamma),
            "lr": float(lr),
            "max_grad_norm": float(max_grad_norm),
            "batch_size": int(batch_size),
            "buffer_capacity": int(buffer_capacity),
            "learning_starts": int(learning_starts),
            "train_frequency": int(train_frequency),
            "target_update_every": int(target_update_every),
            "epsilon_start": float(epsilon_start),
            "epsilon_end": float(epsilon_end),
            "epsilon_decay_steps": int(epsilon_decay_steps),
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

    q_online = LatentQNetwork(latent_dim=bridge.latent_dim, n_actions=n_actions_total)
    q_target = LatentQNetwork(latent_dim=bridge.latent_dim, n_actions=n_actions_total)
    q_online.to(dev)
    q_target.to(dev)
    q_target.load_state_dict(q_online.state_dict())
    q_target.eval()
    optimizer = optim.Adam(q_online.parameters(), lr=float(lr))

    buffer = LatentReplayBuffer(buffer_capacity, bridge.latent_dim, rng)

    enc_dev = bridge.device_t
    pol_dev = next(q_online.parameters()).device
    print("Модели и устройства:")
    print(f"  FastDepth / EnvEncoder --> {enc_dev}")
    print(f"  DQN Q-сеть            --> {pol_dev}")
    if not _same_torch_device(enc_dev, pol_dev):
        print("⚠ Латент и Q-сеть на разных устройствах (редкий случай).")

    rewards = deque(maxlen=100)
    success_window = deque(maxlen=max(5, int(curriculum_window)))
    goal_window = deque(maxlen=max(5, int(curriculum_window)))
    collision_window = deque(maxlen=max(5, int(curriculum_window)))
    timeout_window = deque(maxlen=max(5, int(curriculum_window)))
    out_window = deque(maxlen=max(5, int(curriculum_window)))
    end_dist_window = deque(maxlen=max(5, int(curriculum_window)))
    best_avg_reward = -float("inf")
    best_path = os.path.join(run_dir, "latent_dqn_best.pth")
    no_improve = 0
    total_env_steps = 0
    last_loss: float | None = None
    last_q_mean: float | None = None
    pbar = tqdm(
        range(num_episodes),
        desc="Обучение (DQN+латент)",
        unit="эпизод",
        dynamic_ncols=True,
    )
    for episode in pbar:
        if curriculum is not None:
            stage = curriculum[stage_idx]
            action_grid = _build_action_grid(
                vx_low=float(env.action_space.low[0]),
                vx_high=float(env.action_space.high[0]),
                n_vx_bins=int(n_vx_bins),
                vy_low=float(stage.get("vy_min", vy_min)),
                vy_high=float(stage.get("vy_max", vy_max)),
                n_vy_bins=int(n_vy_bins),
            )

        obs, _ = env.reset(seed=None)
        bridge.reset(env)
        total_reward = 0.0
        steps = 0
        termination_reason = None
        end_distance = None
        sum_min_clearance = 0.0
        n_clearance = 0
        low_clearance_steps = 0

        while steps < max_steps_per_episode:
            z_t = bridge.encode_observation_tensor(obs).to(dev)
            z_np = z_t.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=True)

            eps = float(epsilon_end + (epsilon_start - epsilon_end) * max(
                0.0, 1.0 - min(1.0, total_env_steps / float(epsilon_decay_steps))
            ))
            if float(rng.random()) < eps:
                a_idx = int(rng.integers(0, n_actions_total))
            else:
                q_online.eval()
                with torch.inference_mode():
                    qv = q_online(z_t)
                    a_idx = int(torch.argmax(qv, dim=-1).item())

            action = action_grid[a_idx].astype(np.float32, copy=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
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

            z_next_t = bridge.encode_observation_tensor(next_obs).to(dev)
            z_next_np = z_next_t.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=True)
            buffer.add(z_np, a_idx, float(reward), z_next_np, 1.0 if done else 0.0)

            total_env_steps += 1
            total_reward += float(reward)
            steps += 1

            if (
                total_env_steps >= int(learning_starts)
                and total_env_steps % int(train_frequency) == 0
                and buffer.size >= int(batch_size)
            ):
                bz, ba, br, bzn, bd = buffer.sample(int(batch_size))
                z_b = torch.from_numpy(bz).to(dev)
                a_b = torch.from_numpy(ba).to(dev).long()
                r_b = torch.from_numpy(br).to(dev)
                zn_b = torch.from_numpy(bzn).to(dev)
                d_b = torch.from_numpy(bd).to(dev)

                q_online.train()
                q_all = q_online(z_b)
                q_sa = q_all.gather(1, a_b.view(-1, 1)).squeeze(1)

                with torch.no_grad():
                    q_next_on = q_online(zn_b)
                    a_best = q_next_on.argmax(dim=1, keepdim=True)
                    q_next_tg = q_target(zn_b).gather(1, a_best).squeeze(1)
                    y = r_b + (1.0 - d_b) * float(gamma) * q_next_tg

                loss = F.smooth_l1_loss(q_sa, y)
                if torch.isnan(loss):
                    pass
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(q_online.parameters(), float(max_grad_norm))
                    optimizer.step()
                    last_loss = float(loss.item())
                    last_q_mean = float(q_sa.detach().mean().item())

            if total_env_steps > 0 and total_env_steps % int(target_update_every) == 0:
                q_target.load_state_dict(q_online.state_dict())

            if done:
                break
            obs = next_obs

        eps_logged = float(
            epsilon_end
            + (epsilon_start - epsilon_end)
            * max(0.0, 1.0 - min(1.0, total_env_steps / float(epsilon_decay_steps)))
        )
        ep_stats = {
            "mean_min_clearance": float(sum_min_clearance / max(n_clearance, 1)),
            "low_clearance_frac": float(low_clearance_steps / max(n_clearance, 1)),
        }
        ep_num = episode + 1
        if termination_reason == "goal_reached":
            success_ckpt = os.path.join(run_dir, f"latent_dqn_success_ep{ep_num}.pth")
            torch.save(q_online.state_dict(), success_ckpt)
            tqdm.write(f"✓ Успех (goal): сохранено до update {success_ckpt}")

        rewards.append(total_reward)
        success = 1.0 if termination_reason == "goal_reached" else 0.0
        collision = 1.0 if termination_reason == "collision" else 0.0
        timeout = 1.0 if termination_reason == "timeout" else 0.0
        out_fail = 1.0 if termination_reason in {"out_of_bounds", "out_of_corridor", "fallen"} else 0.0
        success_window.append(success)
        goal_window.append(success)
        collision_window.append(collision)
        timeout_window.append(timeout)
        out_window.append(out_fail)
        if end_distance is not None and np.isfinite(end_distance):
            end_dist_window.append(float(end_distance))

        avg_reward = float(np.mean(rewards))
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
        writer.add_scalar("episode/mean_min_clearance", float(ep_stats["mean_min_clearance"]), ep_num)
        writer.add_scalar("episode/low_clearance_frac", float(ep_stats["low_clearance_frac"]), ep_num)
        writer.add_scalar("schedule/epsilon", eps_logged, ep_num)
        writer.add_scalar("train/total_env_steps", float(total_env_steps), ep_num)
        writer.add_scalar("curriculum/stage_idx", float(stage_idx), ep_num)
        if last_loss is not None:
            writer.add_scalar("loss/dqn", last_loss, ep_num)
        if last_q_mean is not None:
            writer.add_scalar("train/q_mean", last_q_mean, ep_num)

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
                "ε": f"{eps_logged:.3f}",
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
            torch.save(q_online.state_dict(), best_path)
            writer.add_scalar("best/avg_reward", best_avg_reward, ep_num)
            tqdm.write(f"★ Лучшая DQN (avgR={best_avg_reward:.2f}): {best_path}")
            no_improve = 0
        else:
            no_improve += 1

        if (episode + 1) % save_interval == 0:
            pth = os.path.join(run_dir, f"latent_dqn_ep{episode + 1}.pth")
            torch.save(q_online.state_dict(), pth)
            tqdm.write(f"✓ Сохранено: {pth}")

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
    final = os.path.join(run_dir, "latent_dqn_final.pth")
    torch.save(q_online.state_dict(), final)
    writer.close()
    env.close()
    print(f"\n✓ Готово. DQN модель: {final}")
