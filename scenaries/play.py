# scenaries/play.py - инференс EnvEncoder + MCTS / PPO / DQN

import inspect
import os

import numpy as np
import torch

from rl.environment import DroneObstacleAvoidanceEnv
from rl.envencoder_bridge import RLEnvEncoderBridge
from rl.mcts import root_mcts_search
from rl.ppo import LatentPolicyNetwork
from rl.dqn import LatentQNetwork
from utils.reward_config import (
    apply_play_reward_overrides,
    load_reward_config,
    resolve_play_reward_config_path,
)

_PLAY_NO_LIMIT_STEPS = 10**7

_SKIP_TRAIN_ENV_SNAPSHOT_KEYS = frozenset(
    {
        "drone_xy_half_span",
        "sensor_buffer_len",
        "action_space_low",
        "action_space_high",
        "camera_height",
        "camera_width",
        "mj_timestep",
    }
)


def _env_kwargs_from_train_snapshot(snapshot: dict | None) -> dict:
    if not snapshot:
        return {}
    snapshot = dict(snapshot)
    if "obstacle_path_length" not in snapshot and "path_length" in snapshot:
        snapshot["obstacle_path_length"] = snapshot["path_length"]
    sig = inspect.signature(DroneObstacleAvoidanceEnv.__init__)
    valid = set(sig.parameters.keys()) - {"self"}
    out: dict = {}
    int_keys = {"takeoff_steps", "num_obstacles", "max_episode_steps"}
    for k, v in snapshot.items():
        if k == "path_length":
            continue
        if k in _SKIP_TRAIN_ENV_SNAPSHOT_KEYS or k not in valid:
            continue
        if k == "initial_position_range":
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                out[k] = (float(v[0]), float(v[1]))
            continue
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int, float)):
            out[k] = int(v) if k in int_keys else float(v)
        elif isinstance(v, str):
            out[k] = v
    return out


def run_play(
    policy_path,
    encoder_path,
    device="mps",
    episodes=3,
    max_steps=500,
    num_obstacles=50,
    render=True,
    seed=None,
    no_limit=False,
    fastdepth_weights=None,
    n_mcts_simulations=24,
    latent_vx_bins=11,
    latent_vy_bins=5,
    latent_vy_min=-0.2,
    latent_vy_max=0.75,
    max_velocity=3.0,
    forward_velocity=0.25,
    mcts_c_puct=1.5,
    gamma=0.8,
    path_length=5.0,
    map_type="random",
    play_algo="mcts",
    reward_config_path=None,
    run_dir=None,
    train_environment_snapshot=None,
):
    if not os.path.isfile(policy_path):
        raise FileNotFoundError(f"Файл модели не найден: {policy_path}")
    if not os.path.isfile(encoder_path):
        raise FileNotFoundError(f"Веса энкодера не найдены: {encoder_path}")
    if no_limit:
        max_steps = _PLAY_NO_LIMIT_STEPS
    dev = torch.device(
        "cpu" if device == "mps" and not torch.backends.mps.is_available() else device
    )
    if str(dev) == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")
    print("=" * 60)
    print(f"Проигрывание: EnvEncoder + {play_algo.upper()}")
    print("=" * 60)
    wlabel = "Q-сеть (latent_dqn)" if play_algo == "dqn" else "критик+prior(vxxvy)"
    print(f"  Веса ({wlabel}): {policy_path}")
    print(f"  Энкодер среды: {encoder_path}")
    print(f"  Устройство: {dev}")
    if no_limit:
        print(f"  Эпизодов: {episodes}, ограничение по времени: нет (до столкновения или Ctrl+C)")
    else:
        print(f"  Эпизодов: {episodes}, макс. шагов: {max_steps}")
    mcts_bins = int(latent_vx_bins) * int(latent_vy_bins)
    print(
        f"  Препятствия: до {num_obstacles}, path_length={path_length}, map_type={map_type}, "
        f"алгоритм play: {play_algo}, MCTS sims: {n_mcts_simulations}, "
        f"сетка vxxvy: {latent_vx_bins}x{latent_vy_bins}={mcts_bins}, визуализация: {render}"
    )
    if play_algo == "ppo":
        print(
            f"  Среда (как train): max_velocity={max_velocity}, forward_velocity={forward_velocity}; "
            "PPO: категориальный выбор действия (как при обучении)"
        )
    else:
        print(
            f"  Среда (как train): max_velocity={max_velocity}, forward_velocity={forward_velocity}"
        )
    rc_path = resolve_play_reward_config_path(reward_config_path, run_dir, policy_path)
    reward_cfg = load_reward_config(rc_path)
    print(f"  Награды/штрафы: {rc_path}")
    print()

    bridge = RLEnvEncoderBridge(
        encoder_path=encoder_path,
        device=str(dev),
        fastdepth_weights=fastdepth_weights,
    )
    actor_actions = int(latent_vx_bins) * int(latent_vy_bins)
    policy = None
    qnet = None
    if play_algo == "dqn":
        qnet = LatentQNetwork(latent_dim=bridge.latent_dim, n_actions=actor_actions)
        state = torch.load(policy_path, map_location=dev, weights_only=True)
        qnet.load_state_dict(state, strict=True)
        qnet.to(dev)
        qnet.eval()
    else:
        policy = LatentPolicyNetwork(latent_dim=bridge.latent_dim, n_actions=actor_actions)
        state = torch.load(policy_path, map_location=dev, weights_only=True)
        policy.load_state_dict(state, strict=True)
        policy.to(dev)
        policy.eval()

    env_kw = _env_kwargs_from_train_snapshot(train_environment_snapshot)
    env_kw.update(
        {
            "render_mode": "human" if render else None,
            "max_velocity": float(max_velocity),
            "forward_velocity": float(forward_velocity),
            "max_episode_steps": max_steps,
            "num_obstacles": num_obstacles,
            "obstacle_path_length": float(path_length),
            "map_type": map_type,
        }
    )
    env = DroneObstacleAvoidanceEnv(**env_kw)
    apply_play_reward_overrides(env, reward_cfg)
    vx_lo = float(env.action_space.low[0])
    vx_hi = float(env.action_space.high[0])

    def restore_fn(se, so):
        env.restore_from_branching_snapshot(se)
        bridge.unpack_odom_state(so)

    play_rng = np.random.default_rng(int(seed) if seed is not None else None)
    all_rewards = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed if seed is not None else (42 + ep))
        bridge.reset(env)
        total_reward = 0
        step_count = 0
        print(f"Эпизод {ep + 1}/{episodes} старт, позиция: {info['position']}")
        try:
            for _step in range(max_steps):
                z_t = bridge.encode_observation_tensor(obs).to(dev)
                env_snap = env.snapshot_for_branching()
                odom_snap = bridge.pack_odom_state()
                with torch.inference_mode():
                    if play_algo == "dqn":
                        qv = qnet(z_t)
                        logits = qv
                    else:
                        _, logits = policy(z_t)
                if play_algo == "mcts":
                    mcts_out = root_mcts_search(
                        env,
                        bridge,
                        policy,
                        env_snap,
                        odom_snap,
                        vx_low=vx_lo,
                        vx_high=vx_hi,
                        n_vx_bins=int(latent_vx_bins),
                        vy_low=float(latent_vy_min),
                        vy_high=float(latent_vy_max),
                        n_vy_bins=int(latent_vy_bins),
                        n_simulations=n_mcts_simulations,
                        gamma=gamma,
                        c_puct=mcts_c_puct,
                        device=dev,
                        prior_logits=logits,
                        restore_fn=restore_fn,
                    )
                    action_vx = float(mcts_out.action_vx)
                    action_vy = float(mcts_out.action_vy)
                elif play_algo == "dqn":
                    q1d = logits.squeeze(0)
                    a_idx = int(torch.argmax(q1d, dim=0).item())
                else:
                    logits_1d = logits.squeeze(0)
                    probs = torch.softmax(logits_1d, dim=0).float().cpu().numpy()
                    probs = probs / max(float(probs.sum()), 1e-12)
                    a_idx = int(play_rng.choice(probs.size, p=probs))
                if play_algo != "mcts":
                    vx_centers = np.linspace(vx_lo, vx_hi, int(latent_vx_bins), dtype=np.float32)
                    vy_centers = np.linspace(float(latent_vy_min), float(latent_vy_max), int(latent_vy_bins), dtype=np.float32)
                    vx_idx = int(a_idx % int(latent_vx_bins))
                    vy_idx = int(a_idx // int(latent_vx_bins))
                    action_vx = float(vx_centers[vx_idx])
                    action_vy = float(vy_centers[min(vy_idx, len(vy_centers) - 1)])
                action = np.array([action_vx, action_vy], dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                if render:
                    env.render()
                total_reward += reward
                step_count += 1
                if terminated or truncated:
                    reason = "столкновение" if terminated else "лимит шагов"
                    print(f"  Эпизод {ep + 1} завершён на шаге {step_count} ({reason}), награда: {total_reward:.2f}")
                    break
            else:
                if not no_limit:
                    print(f"  Эпизод {ep + 1} завершён по шагам ({step_count}), награда: {total_reward:.2f}")
        except KeyboardInterrupt:
            print("\nОстановлено пользователем")
            break
        all_rewards.append(total_reward)
    env.close()
    if all_rewards:
        print(f"\nИтог: средняя награда по эпизодам: {np.mean(all_rewards):.2f}")
    print("Environment закрыт.")
