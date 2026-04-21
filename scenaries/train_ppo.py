# scenaries/train_ppo.py - обучение PPO в латенте

import os
from collections import deque
import typing

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rl.environment import DroneObstacleAvoidanceEnv
from rl.envencoder_bridge import RLEnvEncoderBridge
from rl.ppo import LatentPolicyNetwork
from rl.ppo import PPOLatentTrainer, _build_action_grid
from utils.reward_config import (
    apply_curriculum_stage_full,
    apply_env_reward_dict,
    build_curriculum_stages,
    dqn_gamma_from_reward_cfg,
    load_reward_config,
    merge_map_train_args_from_reward_yaml,
    merge_ppo_train_args_from_reward_yaml,
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


def train(
    num_episodes=1000,
    max_steps_per_episode=1000,
    save_interval=100,
    model_dir="models/ppo-latent",
    log_dir=None,
    device="mps",
    num_obstacles=5,
    encoder_path: str | None = None,
    fastdepth_weights=None,
    n_vx_bins=11,
    n_vy_bins=5,
    vy_min: float = -0.2,
    vy_max: float = 0.75,
    policy_updates: int = 4,
    gamma: typing.Optional[float] = None,
    gae_lambda: float = 0.95,
    ppo_clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    lr: float = 3e-4,
    max_grad_norm: float = 0.5,
    seed: int | None = 42,
    path_length: float = 5.0,
    map_type: str = "random",
    early_stop_patience: int = 20,
    early_stop_min_episodes: int = 30,
    entropy_coef_final: float = 0.003,
    entropy_decay_fraction: float = 0.6,
    use_curriculum: bool = True,
    curriculum_success_threshold: typing.Optional[float] = None,
    curriculum_window: typing.Optional[int] = None,
    curriculum_min_episodes_per_stage: typing.Optional[int] = None,
    curriculum_use_fallback: bool = True,
    curriculum_fallback_max_mean_dist: float = 0.55,
    curriculum_fallback_max_collision_rate: float = 0.38,
    entropy_floor_until_goal: float = 0.012,
    entropy_high_out_rate_threshold: float = 0.5,
    entropy_floor_when_high_out_rate: float = 0.02,
    reward_config_path: str | None = None,
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

    merged = merge_ppo_train_args_from_reward_yaml(
        reward_cfg,
        {
            "gamma": gamma,
            "policy_updates": policy_updates,
            "gae_lambda": gae_lambda,
            "ppo_clip_eps": ppo_clip_eps,
            "entropy_coef": entropy_coef,
            "entropy_coef_final": entropy_coef_final,
            "entropy_decay_fraction": entropy_decay_fraction,
            "value_coef": value_coef,
            "lr": lr,
            "max_grad_norm": max_grad_norm,
            "n_vx_bins": n_vx_bins,
            "n_vy_bins": n_vy_bins,
            "vy_min": vy_min,
            "vy_max": vy_max,
            "use_curriculum": use_curriculum,
            "curriculum_success_threshold": curriculum_success_threshold,
            "curriculum_window": curriculum_window,
            "curriculum_min_episodes_per_stage": curriculum_min_episodes_per_stage,
            "curriculum_use_fallback": curriculum_use_fallback,
            "curriculum_fallback_max_mean_dist": curriculum_fallback_max_mean_dist,
            "curriculum_fallback_max_collision_rate": curriculum_fallback_max_collision_rate,
            "early_stop_patience": early_stop_patience,
            "early_stop_min_episodes": early_stop_min_episodes,
            "entropy_floor_until_goal": entropy_floor_until_goal,
            "entropy_high_out_rate_threshold": entropy_high_out_rate_threshold,
            "entropy_floor_when_high_out_rate": entropy_floor_when_high_out_rate,
        },
    )
    gamma = merged["gamma"]
    policy_updates = merged["policy_updates"]
    gae_lambda = merged["gae_lambda"]
    ppo_clip_eps = merged["ppo_clip_eps"]
    entropy_coef = merged["entropy_coef"]
    entropy_coef_final = merged["entropy_coef_final"]
    entropy_decay_fraction = merged["entropy_decay_fraction"]
    value_coef = merged["value_coef"]
    lr = merged["lr"]
    max_grad_norm = merged["max_grad_norm"]
    n_vx_bins = merged["n_vx_bins"]
    n_vy_bins = merged["n_vy_bins"]
    vy_min = merged["vy_min"]
    vy_max = merged["vy_max"]
    use_curriculum = bool(merged["use_curriculum"])
    curriculum_success_threshold = merged["curriculum_success_threshold"]
    curriculum_window = merged["curriculum_window"]
    curriculum_min_episodes_per_stage = merged["curriculum_min_episodes_per_stage"]
    curriculum_use_fallback = merged["curriculum_use_fallback"]
    curriculum_fallback_max_mean_dist = merged["curriculum_fallback_max_mean_dist"]
    curriculum_fallback_max_collision_rate = merged["curriculum_fallback_max_collision_rate"]
    early_stop_patience = merged["early_stop_patience"]
    early_stop_min_episodes = merged["early_stop_min_episodes"]
    entropy_floor_until_goal = merged["entropy_floor_until_goal"]
    entropy_high_out_rate_threshold = merged["entropy_high_out_rate_threshold"]
    entropy_floor_when_high_out_rate = merged["entropy_floor_when_high_out_rate"]
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
        gy = dqn_gamma_from_reward_cfg(reward_cfg)
        if gy is not None:
            gamma = gy
    if gamma is None:
        gamma = 0.8
    gamma = float(gamma)

    if log_dir is None:
        log_dir = os.path.join(run_dir, "tensorboard")
    log_dir = os.path.abspath(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    print(f"✓ PPO запуск: {run_dir}")
    print(f"✓ Копия наград: {reward_copy_path}")
    print(f"✓ TensorBoard: {log_dir}")
    print(
        f"PPO: updates={policy_updates}, clip={ppo_clip_eps}, entropy={entropy_coef}, "
        f"γ={gamma}, λ_GAE={gae_lambda}, actions={n_vx_bins}x{n_vy_bins}, "
        f"reward_config={reward_yaml_resolved}"
    )
    if not use_curriculum:
        print("✓ Curriculum выключен (CLI или ppo.use_curriculum в YAML наград)")

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
    n_actions_total = int(n_vx_bins) * int(n_vy_bins)
    policy = LatentPolicyNetwork(latent_dim=bridge.latent_dim, n_actions=n_actions_total)
    trainer = PPOLatentTrainer(
        env=env,
        bridge=bridge,
        policy_network=policy,
        device=device,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        max_grad_norm=max_grad_norm,
        ppo_clip_eps=ppo_clip_eps,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        policy_updates=policy_updates,
        n_vx_bins=n_vx_bins,
        n_vy_bins=n_vy_bins,
        vx_low=float(env.action_space.low[0]),
        vx_high=float(env.action_space.high[0]),
        vy_low=float(vy_min),
        vy_high=float(vy_max),
        seed=seed,
    )

    run_cfg_path = write_train_run_config_yaml(
        run_dir=run_dir,
        algorithm="ppo",
        env=env,
        bridge=bridge,
        encoder_path=encoder_path,
        device=device,
        seed=seed,
        policy_inference={
            "network": "LatentPolicyNetwork",
            "latent_dim": int(bridge.latent_dim),
            "n_actions": int(n_actions_total),
            "discretization": {
                "n_vx_bins": int(trainer.n_vx_bins),
                "n_vy_bins": int(trainer.n_vy_bins),
                "vy_min": float(vy_min),
                "vy_max": float(vy_max),
                "vx_low": float(env.action_space.low[0]),
                "vx_high": float(env.action_space.high[0]),
            },
        },
        training_hyperparams={
            "lr": float(trainer.optimizer.param_groups[0]["lr"]),
            "gamma": float(trainer.gamma),
            "gae_lambda": float(trainer.gae_lambda),
            "ppo_clip_eps": float(trainer.ppo_clip_eps),
            "entropy_coef_start": float(entropy_coef),
            "entropy_coef_final": float(entropy_coef_final),
            "entropy_decay_fraction": float(entropy_decay_fraction),
            "entropy_floor_until_goal": float(entropy_floor_until_goal),
            "entropy_high_out_rate_threshold": float(entropy_high_out_rate_threshold),
            "entropy_floor_when_high_out_rate": float(entropy_floor_when_high_out_rate),
            "value_coef": float(trainer.value_coef),
            "max_grad_norm": float(trainer.max_grad_norm),
            "policy_updates": int(trainer.policy_updates),
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
    pol_dev = trainer.policy_device
    print("Модели и устройства:")
    print(f"  FastDepth (глубина RGB-->depth) --> {enc_dev}")
    print(f"  EnvEncoder (depth+odom-->z)     --> {enc_dev}")
    print(f"  PPO actor-critic               --> {pol_dev}")
    if not _same_torch_device(enc_dev, pol_dev):
        print("⚠ Латент и политика на разных устройствах (редкий случай).")

    rewards = deque(maxlen=100)
    success_window = deque(maxlen=max(5, int(curriculum_window)))
    goal_window = deque(maxlen=max(5, int(curriculum_window)))
    collision_window = deque(maxlen=max(5, int(curriculum_window)))
    timeout_window = deque(maxlen=max(5, int(curriculum_window)))
    out_window = deque(maxlen=max(5, int(curriculum_window)))
    out_bounds_window = deque(maxlen=max(5, int(curriculum_window)))
    out_corridor_window = deque(maxlen=max(5, int(curriculum_window)))
    fallen_window = deque(maxlen=max(5, int(curriculum_window)))
    end_dist_window = deque(maxlen=max(5, int(curriculum_window)))
    best_avg_reward = -float("inf")
    best_path = os.path.join(run_dir, "latent_policy_best.pth")
    no_improve = 0
    saw_any_goal = False
    total_env_steps = 0
    pbar = tqdm(
        range(num_episodes),
        desc="Обучение (PPO+латент)",
        unit="эпизод",
        dynamic_ncols=True,
    )
    for episode in pbar:
        if entropy_decay_fraction <= 0.0:
            entropy_scheduled = float(entropy_coef_final)
        else:
            t = min(1.0, float(episode) / max(1.0, float(num_episodes) * float(entropy_decay_fraction)))
            entropy_scheduled = float(entropy_coef + t * (float(entropy_coef_final) - float(entropy_coef)))
        if not saw_any_goal:
            entropy_now = max(float(entropy_scheduled), float(entropy_floor_until_goal))
        else:
            entropy_now = float(entropy_scheduled)
        if curriculum is not None:
            stage = curriculum[stage_idx]
            trainer.action_grid = _build_action_grid(
                vx_low=float(env.action_space.low[0]),
                vx_high=float(env.action_space.high[0]),
                n_vx_bins=trainer.n_vx_bins,
                vy_low=float(stage.get("vy_min", vy_min)),
                vy_high=float(stage.get("vy_max", vy_max)),
                n_vy_bins=trainer.n_vy_bins,
            )
        total_reward, steps, termination_reason, end_distance, ep_stats = trainer.collect_trajectory(
            max_steps=max_steps_per_episode, seed=None
        )
        total_env_steps += int(steps)
        ep_num = episode + 1
        if termination_reason == "goal_reached":
            success_ckpt = os.path.join(run_dir, f"latent_policy_success_ep{ep_num}.pth")
            trainer.save_policy(success_ckpt)
            tqdm.write(f"✓ Успех (goal): сохранено до PPO-update {success_ckpt}")

        rewards.append(total_reward)
        success = 1.0 if termination_reason == "goal_reached" else 0.0
        if success > 0.0:
            saw_any_goal = True
        goal = success
        collision = 1.0 if termination_reason == "collision" else 0.0
        timeout = 1.0 if termination_reason == "timeout" else 0.0
        out_fail = 1.0 if termination_reason in {"out_of_bounds", "out_of_corridor", "fallen"} else 0.0
        out_bounds = 1.0 if termination_reason == "out_of_bounds" else 0.0
        out_corridor = 1.0 if termination_reason == "out_of_corridor" else 0.0
        fallen = 1.0 if termination_reason == "fallen" else 0.0
        term_unknown = 1.0 if termination_reason is None else 0.0
        success_window.append(success)
        goal_window.append(goal)
        collision_window.append(collision)
        timeout_window.append(timeout)
        out_window.append(out_fail)
        out_bounds_window.append(out_bounds)
        out_corridor_window.append(out_corridor)
        fallen_window.append(fallen)
        if end_distance is not None and np.isfinite(end_distance):
            end_dist_window.append(float(end_distance))
        gs = ep_num
        avg_reward = float(np.mean(rewards))
        success_rate = float(np.mean(success_window)) if success_window else 0.0
        goal_rate = float(np.mean(goal_window)) if goal_window else 0.0
        collision_rate = float(np.mean(collision_window)) if collision_window else 0.0
        timeout_rate = float(np.mean(timeout_window)) if timeout_window else 0.0
        out_rate = float(np.mean(out_window)) if out_window else 0.0
        out_bounds_rate = float(np.mean(out_bounds_window)) if out_bounds_window else 0.0
        out_corridor_rate = float(np.mean(out_corridor_window)) if out_corridor_window else 0.0
        fallen_rate = float(np.mean(fallen_window)) if fallen_window else 0.0
        mean_end_dist = float(np.mean(end_dist_window)) if end_dist_window else float("nan")

        entropy_for_update = float(entropy_now)
        if out_rate > float(entropy_high_out_rate_threshold):
            entropy_for_update = max(
                entropy_for_update, float(entropy_floor_when_high_out_rate)
            )

        adv, rets = trainer.compute_gae()
        metrics = trainer.update(adv, rets, entropy_coef=entropy_for_update)
        writer.add_scalar("episode/reward", total_reward, gs)
        writer.add_scalar("episode/length", steps, gs)
        writer.add_scalar("episode/avg_reward", avg_reward, gs)
        writer.add_scalar("episode/success", success, gs)
        writer.add_scalar("episode/success_rate_window", success_rate, gs)
        writer.add_scalar("episode/goal_rate_window", goal_rate, gs)
        writer.add_scalar("episode/collision_rate_window", collision_rate, gs)
        writer.add_scalar("episode/timeout_rate_window", timeout_rate, gs)
        writer.add_scalar("episode/out_rate_window", out_rate, gs)
        writer.add_scalar("episode/out_bounds_rate_window", out_bounds_rate, gs)
        writer.add_scalar("episode/out_corridor_rate_window", out_corridor_rate, gs)
        writer.add_scalar("episode/fallen_rate_window", fallen_rate, gs)
        writer.add_scalar("episode/termination_goal", success, gs)
        writer.add_scalar("episode/termination_collision", collision, gs)
        writer.add_scalar("episode/termination_timeout", timeout, gs)
        writer.add_scalar("episode/termination_out_bounds", out_bounds, gs)
        writer.add_scalar("episode/termination_out_corridor", out_corridor, gs)
        writer.add_scalar("episode/termination_fallen", fallen, gs)
        writer.add_scalar("episode/termination_unknown", term_unknown, gs)
        if np.isfinite(mean_end_dist):
            writer.add_scalar("episode/mean_dist_to_goal_end", mean_end_dist, gs)
        writer.add_scalar("episode/mean_min_clearance", float(ep_stats["mean_min_clearance"]), gs)
        writer.add_scalar("episode/low_clearance_frac", float(ep_stats["low_clearance_frac"]), gs)
        rtm = ep_stats.get("reward_term_means") or {}
        if isinstance(rtm, dict):
            for rk, rv in rtm.items():
                if not isinstance(rk, str) or not isinstance(rv, (int, float)):
                    continue
                if not np.isfinite(float(rv)):
                    continue
                safe = rk.replace("/", "_")
                writer.add_scalar(f"reward_terms/mean_per_step/{safe}", float(rv), gs)
        writer.add_scalar("schedule/entropy_coef", entropy_for_update, gs)
        writer.add_scalar("schedule/entropy_scheduled", float(entropy_scheduled), gs)
        writer.add_scalar("schedule/gamma", float(gamma), gs)
        writer.add_scalar("train/saw_any_goal", 1.0 if saw_any_goal else 0.0, gs)
        writer.add_scalar("train/total_env_steps", float(total_env_steps), gs)
        writer.add_scalar("curriculum/stage_idx", float(stage_idx), gs)
        if curriculum is not None:
            st0 = curriculum[stage_idx]
            writer.add_scalar("curriculum/path_length", float(env.obstacle_path_length), gs)
            fb_coll0 = float(
                st0.get("fallback_max_collision_rate", curriculum_fallback_max_collision_rate)
            )
            fb_dist0 = float(
                st0.get("fallback_max_mean_dist", curriculum_fallback_max_mean_dist)
            )
            frd0 = st0.get("fallback_relaxed_mean_dist")
            fro0 = float(st0.get("fallback_max_out_rate", 0.4))
            writer.add_scalar("curriculum/threshold_success_rate", float(curriculum_success_threshold), gs)
            writer.add_scalar(
                "curriculum/gap_success_vs_threshold",
                success_rate - float(curriculum_success_threshold),
                gs,
            )
            writer.add_scalar("curriculum/gap_collision_headroom", fb_coll0 - collision_rate, gs)
            writer.add_scalar("curriculum/gap_out_headroom", fro0 - out_rate, gs)
            if np.isfinite(mean_end_dist):
                writer.add_scalar(
                    "curriculum/gap_strict_dist_headroom",
                    fb_dist0 - mean_end_dist,
                    gs,
                )
                if frd0 is not None:
                    writer.add_scalar(
                        "curriculum/gap_relaxed_dist_headroom",
                        float(frd0) - mean_end_dist,
                        gs,
                    )
        if metrics["policy_loss"] is not None:
            writer.add_scalar("loss/policy", metrics["policy_loss"], gs)
        if metrics["value_loss"] is not None:
            writer.add_scalar("loss/value", metrics["value_loss"], gs)
        if metrics["entropy"] is not None:
            writer.add_scalar("loss/entropy", metrics["entropy"], gs)
        pbar.set_postfix(
            {
                "R": f"{total_reward:.2f}",
                "avgR": f"{avg_reward:.2f}",
                "succ": f"{success_rate:.2f}",
                "coll": f"{collision_rate:.2f}",
                "tout": f"{timeout_rate:.2f}",
                "stage": f"{stage_idx + 1}",
                "steps": f"{steps}",
            }
        )

        if curriculum is not None and stage_idx < (len(curriculum) - 1):
            stage_episodes = gs - stage_start_episode + 1
            st_cur = curriculum[stage_idx]
            fb_dist = float(
                st_cur.get("fallback_max_mean_dist", curriculum_fallback_max_mean_dist)
            )
            fb_coll = float(
                st_cur.get("fallback_max_collision_rate", curriculum_fallback_max_collision_rate)
            )
            win_ok = len(success_window) >= max(5, int(curriculum_window))
            ep_ok = stage_episodes >= int(curriculum_min_episodes_per_stage)
            by_success = (
                win_ok
                and ep_ok
                and success_rate >= float(curriculum_success_threshold)
            )
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
                stage_start_episode = gs + 1
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
                writer.add_scalar("curriculum/advance_reason", adv_code, gs)
                tqdm.write(
                    f"⇧ Curriculum stage {stage_idx + 1}/{len(curriculum)}: {curriculum[stage_idx]['name']} "
                    f"({adv_tag}; succ={success_rate:.2f}, mean_dist={mean_end_dist:.3f}, "
                    f"coll={collision_rate:.2f}, out={out_rate:.2f})"
                )

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            trainer.save_policy(best_path)
            writer.add_scalar("best/avg_reward", best_avg_reward, gs)
            tqdm.write(f"★ Лучшая PPO модель (avgR={best_avg_reward:.2f}): {best_path}")
            no_improve = 0
        else:
            no_improve += 1

        if (episode + 1) % save_interval == 0:
            pth = os.path.join(run_dir, f"latent_policy_ep{episode + 1}.pth")
            trainer.save_policy(pth)
            tqdm.write(f"✓ Сохранено: {pth}")

        if device == "mps" and (episode + 1) % 25 == 0 and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if (
            gs >= int(early_stop_min_episodes)
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
    trainer.save_policy(final)
    writer.close()
    env.close()
    print(f"\n✓ Готово. PPO модель: {final}")
