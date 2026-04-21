# scenaries/random_policy.py - случайная политика в среде

import numpy as np

from rl.environment import DroneObstacleAvoidanceEnv


def run_random_policy(
    render_mode="human",
    steps=500,
    num_obstacles=50,
    episodes=1,
    seed=None,
):
    print("=" * 60)
    print("Сценарий случайного перемещения")
    print("=" * 60)
    print(f"  Эпизодов: {episodes}, макс. шагов за эпизод: {steps}")
    print(f"  Препятствия: до {num_obstacles}, визуализация: {render_mode == 'human'}")
    print()

    env = DroneObstacleAvoidanceEnv(
        render_mode=render_mode,
        max_velocity=0.4,
        forward_velocity=0.25,
        max_episode_steps=steps,
        num_obstacles=num_obstacles,
        obstacle_path_length=5.0,
    )

    all_rewards = []
    all_lengths = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed if seed is not None else (42 + ep))
        total_reward = 0
        step_count = 0
        print(f"Эпизод {ep + 1}/{episodes} старт, позиция: {info['position']}")
        try:
            for _ in range(steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if render_mode == "human":
                    env.render()
                total_reward += reward
                step_count += 1
                if terminated or truncated:
                    reason = "столкновение" if terminated else "лимит шагов"
                    print(f"  Эпизод {ep + 1} завершён на шаге {step_count} ({reason}), награда: {total_reward:.2f}")
                    break
            else:
                print(f"  Эпизод {ep + 1} завершён по шагам ({step_count}), награда: {total_reward:.2f}")
        except KeyboardInterrupt:
            print("\nОстановлено пользователем")
            break
        all_rewards.append(total_reward)
        all_lengths.append(step_count)

    env.close()
    if all_rewards:
        print(f"\nИтог: средняя награда: {np.mean(all_rewards):.2f}, средняя длина эпизода: {np.mean(all_lengths):.0f} шагов")
    print("Environment закрыт.")
