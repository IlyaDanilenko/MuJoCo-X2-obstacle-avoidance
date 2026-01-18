# main.py - скрипт запуска сценария обучения

import argparse
import numpy as np
from rl.environment import DroneObstacleAvoidanceEnv

def view_environment(render_mode="human", steps=1000, num_obstacles=100):
    print("=" * 60)
    print("Создание environment для просмотра")
    print("=" * 60)
    
    # Создаем environment с препятствиями
    env = DroneObstacleAvoidanceEnv(
        render_mode=render_mode,
        max_velocity=2.0,
        forward_velocity=0.0,
        image_size=(64, 64),
        max_episode_steps=steps,
        num_obstacles=num_obstacles,
    )
    
    print("Environment создан")
    print(f"  Режим рендеринга: {render_mode}")
    print(f"  Максимальное количество шагов: {steps}")
    print(f"  Максимальное количество препятствий: {num_obstacles}")
    
    # Сбрасываем environment
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
        for step in range(steps):
            # Простое действие: летим вперед без бокового смещения
            action = np.array([0.0, 0.0])
            
            # Выполняем шаг
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Обновляем визуализацию
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


def run_random_policy(render_mode="human", steps=500, num_obstacles=50):
    print("Выбрана: Случайная политика")
    print("=" * 60)
    
    env = DroneObstacleAvoidanceEnv(
        render_mode=render_mode,
        max_velocity=2.0,
        image_size=(64, 64),
        max_episode_steps=steps,
        num_obstacles=num_obstacles,
    )
    
    obs, info = env.reset(seed=42)
    total_reward = 0
    
    for step in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if render_mode == "human":
            env.render()
        
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"Эпизод завершен. Общая награда: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    DEFAULT_MAX_OBSTACLE = 50

    parser = argparse.ArgumentParser()
    
    subparsers = parser.add_subparsers(dest='command', help='Команда для выполнения')
    
    # Парсер для случайной политики
    random_parser = subparsers.add_parser('random', help='Запуск со случайной политикой')
    random_parser.add_argument('--render', action='store_true', default=False,
                             help='Включить визуализацию')
    random_parser.add_argument('--steps', type=int, default=500, help='Количество шагов')
    random_parser.add_argument('--num-obstacles', type=int, default=DEFAULT_MAX_OBSTACLE,
                             help='Максимальное количество препятствий (по умолчанию 5)')
    
    # Парсер для просмотра environment
    view_parser = subparsers.add_parser('view', help='Просмотр environment с препятствиями')
    view_parser.add_argument('--render', action='store_true', default=True,
                           help='Включить визуализацию (по умолчанию включена)')
    view_parser.add_argument('--no-render', dest='render', action='store_false',
                            help='Отключить визуализацию')
    view_parser.add_argument('--steps', type=int, default=1000, help='Количество шагов симуляции')
    view_parser.add_argument('--num-obstacles', type=int, default=DEFAULT_MAX_OBSTACLE,
                            help='Максимальное количество препятствий (по умолчанию 100)')
    
    args = parser.parse_args()
    
    if args.command == 'random':
        env_render_mode = "human" if args.render else None
        run_random_policy(env_render_mode, args.steps, args.num_obstacles)
    elif args.command == 'view':
        env_render_mode = "human" if args.render else None
        view_environment(env_render_mode, args.steps, args.num_obstacles)
