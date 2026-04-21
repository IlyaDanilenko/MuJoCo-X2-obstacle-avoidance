# scenaries/view_environment.py - просмотр среды с препятствиями

import numpy as np

from rl.environment import DroneObstacleAvoidanceEnv

_PLAY_NO_LIMIT_STEPS = 10**7


def view_environment(
    render_mode="human",
    steps=1000,
    num_obstacles=100,
    no_limit=False,
    path_length=5.0,
    map_type="random",
):
    print("=" * 60)
    print("Создание environment для просмотра")
    print("=" * 60)

    cap = _PLAY_NO_LIMIT_STEPS if no_limit else steps

    env = DroneObstacleAvoidanceEnv(
        render_mode=render_mode,
        max_velocity=0.4,
        forward_velocity=0.0,
        max_episode_steps=cap,
        num_obstacles=num_obstacles,
        obstacle_path_length=path_length,
        map_type=map_type,
    )

    print("Environment создан")
    print(f"  Режим рендеринга: {render_mode}")
    if no_limit:
        print("  Максимальное количество шагов: без лимита (до столкновения/границы или Ctrl+C)")
    else:
        print(f"  Максимальное количество шагов: {steps}")
    print(f"  Максимальное количество препятствий: {num_obstacles}")
    print(f"  Длина пути генерации препятствий: {path_length}")
    print(f"  Тип карты: {map_type}")

    obs, info = env.reset(seed=42)

    print("\nНачальные наблюдения:")
    print(f"  Изображение: {obs['image'].shape}")
    print(f"  Акселерометр: {obs['accelerometer']}")
    print(f"  Гироскоп: {obs['gyroscope']}")
    print(f"  Позиция дрона: {info['position']}")

    print("\n" + "=" * 60)
    print("Запуск симуляции...")

    total_reward = 0
    step_count = 0

    try:
        for step in range(cap):
            action = np.array([0.0], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)

            if env.render_mode == "human":
                env.render()

            total_reward += reward
            step_count += 1

            if step % 100 == 0:
                print(f"Шаг {step}: Позиция={info['position']}, Награда={reward:.2f}")

            if terminated or truncated:
                print(f"\nЭпизод завершен на шаге {step}")
                print(f"  Причина: {'Столкновение' if terminated else 'Превышено время'}")
                break
    except KeyboardInterrupt:
        print("\n\nСимуляция остановлена")

    print(f"\nИтоги:")
    print(f"  Всего шагов: {step_count}")
    print(f"  Общая награда: {total_reward:.2f}")

    env.close()
    print("\nEnvironment закрыт")
