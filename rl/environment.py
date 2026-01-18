# rl/environment.py - код среды обучения

import numpy as np
import mujoco
import cv2
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
from sim.scene import Scene
from sim.uav import Drone
from rl.map_generator import generate_map_around_path


class DroneObstacleAvoidanceEnv(gym.Env):
    """
    Gymnasium environment для обучения БПЛА избеганию препятствий.
    
    Observation:
        - Изображение с камеры БПЛА (H, W, 3)
        - Данные акселерометра (3,) - линейное ускорение
        - Данные гироскопа (3,) - угловая скорость
    
    Action:
        - Скорости: [vy, vz] где:
          vy: скорость влево/вправо (-max_velocity до max_velocity)
          vz: скорость вверх/вниз (-max_velocity до max_velocity)
          vx (скорость вперед) фиксирована и всегда положительна (forward_velocity)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_velocity: float = 2.0,
        forward_velocity: float = 1.0,
        image_size: Tuple[int, int] = (64, 64),
        max_episode_steps: int = 1000,
        collision_distance: float = 0.5,
        reward_collision: float = -100.0,
        reward_survival: float = 0.1,
        reward_progress: float = 1.0,
        initial_position_range: Tuple[float, float] = (-2.0, 2.0),
        initial_height: float = 1.0,
        takeoff_steps: int = 100,
        num_obstacles: int = 5,  # Используется как максимальное количество (реальное определяется шумом)
        obstacle_path_length: float = 5.0,
        min_passage_width: float = 1.5,  # Минимальная ширина прохода между столбами
    ):
        """
        Инициализация environment.
        
        Args:
            render_mode: Режим рендеринга ('human' или 'rgb_array')
            max_velocity: Максимальная скорость дрона для бокового движения и высоты
            forward_velocity: Фиксированная скорость вперед (всегда положительная)
            image_size: Размер изображения с камеры (height, width)
            max_episode_steps: Максимальное количество шагов в эпизоде
            collision_distance: Расстояние до препятствия, считающееся столкновением
            reward_collision: Награда за столкновение
            reward_survival: Награда за каждый шаг без столкновения
            reward_progress: Награда за движение вперед
            initial_position_range: Диапазон начальных позиций (x, y)
            initial_height: Целевая высота дрона после взлета
            takeoff_steps: Количество шагов для фазы взлета (по умолчанию 100)
            num_obstacles: Максимальное количество препятствий (столбов) в сцене (по умолчанию 5)
                          Реальное количество определяется шумом
            obstacle_path_length: Длина пути, на котором размещаются препятствия (по умолчанию 10.0)
            min_passage_width: Минимальная ширина прохода между столбами (по умолчанию 1.5)
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.max_velocity = max_velocity
        self.forward_velocity = forward_velocity  # Фиксированная скорость вперед
        self.image_size = image_size
        self.max_episode_steps = max_episode_steps
        self.collision_distance = collision_distance
        self.reward_collision = reward_collision
        self.reward_survival = reward_survival
        self.reward_progress = reward_progress
        self.initial_position_range = initial_position_range
        self.initial_height = initial_height
        self.takeoff_steps = takeoff_steps  # Количество шагов для взлета
        self.num_obstacles = num_obstacles  # Максимальное количество столбов
        self.obstacle_path_length = obstacle_path_length
        self.min_passage_width = min_passage_width  # Минимальная ширина прохода

        self.scene = None
        self.drone = None
        
        # Инициализируем сцену и дрон с начальными препятствиями
        self._initialize_scene_and_drone(initial_height, max_velocity, 0.0, 0.0)
        
        # Определяем observation space
        # Изображение: (H, W, 3) в диапазоне [0, 255]
        # Акселерометр: (3,) - линейное ускорение
        # Гироскоп: (3,) - угловая скорость
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255, 
                shape=(image_size[0], image_size[1], 3), 
                dtype=np.uint8
            ),
            "accelerometer": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(3,), 
                dtype=np.float32
            ),
            "gyroscope": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(3,), 
                dtype=np.float32
            ),
        })
        
        # Определяем action space: только vy
        # vy - скорость влево/вправо
        # vx (скорость вперед) фиксирована и всегда положительна
        self.action_space = spaces.Box(
            low=-max_velocity,
            high=max_velocity,
            shape=(2,),
            dtype=np.float32
        )
        
        # Счетчик шагов
        self.step_count = 0
        self.prev_position = None
        
        self.viewer = None
        
    def _initialize_scene_and_drone(self, initial_height, max_velocity, start_x=0.0, start_y=0.0):
        """
        Инициализирует сцену с препятствиями и дрон
        
        Args:
            initial_height: Начальная высота дрона
            max_velocity: Максимальная скорость дрона
            start_x: Начальная X координата дрона (для размещения препятствий)
            start_y: Начальная Y координата дрона (для размещения препятствий)
        """
        # Создаем сцену
        self.scene = Scene()
        
        # Генерируем карту столбов с использованием генератора карты
        # Генерируем случайное семя для шума в этом эпизоде
        noise_seed = np.random.randint(0, 10000)
        
        # Генерируем столбы в круговой области вокруг пути движения
        pillars = generate_map_around_path(
            start_x=start_x,
            start_y=start_y,
            path_length=self.obstacle_path_length,
            min_pillars=3,
            max_pillars=self.num_obstacles,
            min_radius=0.3,
            max_radius=0.8,
            min_height=1.0,
            max_height=6.0,
            min_passage_width=self.min_passage_width,
            seed=noise_seed,
        )
        
        # Добавляем столбы в сцену
        for i, pillar in enumerate(pillars):
            self.scene.add_pillar(
                x=pillar.x,
                y=pillar.y,
                radius=pillar.radius,
                height=pillar.height,
                name=f"pillar_{i+1}"
            )
        
        self.drone = Drone(target=np.array((0, 0, initial_height)), scene=self.scene)
        self.drone.max_velocity = max_velocity
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        Получает текущие наблюдения.
        
        Returns:
            Словарь с наблюдениями: image, accelerometer, gyroscope
        """
        # Получаем изображение с камеры
        camera_image = self.drone.get_camera_image()
        # Изменяем размер изображения
        image_resized = cv2.resize(camera_image, self.image_size)
        
        # Получаем данные сенсоров через методы класса Drone
        accel_data = self.drone.get_accelerometer_data()
        gyro_data = self.drone.get_gyroscope_data()
        
        return {
            "image": image_resized,
            "accelerometer": accel_data,
            "gyroscope": gyro_data,
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Возвращает дополнительную информацию для отладки"""
        position = self.drone.sensor.get_position()[:3]
        velocity = self.drone.sensor.get_velocity()[:3]
        
        return {
            "position": position,
            "velocity": velocity,
            "step": self.step_count,
        }
    
    def _check_collision(self) -> bool:
        """
        Проверяет столкновение дрона с препятствиями.
        Любая часть дрона, которая столкнулась с препятствием, засчитывается.
        
        Returns:
            True если произошло столкновение
        """
        # Получаем ID body дрона для проверки принадлежности геометрий
        drone_body_id = mujoco.mj_name2id(
            self.drone.m, mujoco.mjtObj.mjOBJ_BODY, "x2"
        )
        
        # Проверяем контакты в MuJoCo
        n_contact = self.drone.d.ncon
        if n_contact > 0:
            for i in range(n_contact):
                contact = self.drone.d.contact[i]
                geom1 = contact.geom1
                geom2 = contact.geom2
                
                # Получаем имена геометрий
                geom1_name = mujoco.mj_id2name(
                    self.drone.m, mujoco.mjtObj.mjOBJ_GEOM, geom1
                )
                geom2_name = mujoco.mj_id2name(
                    self.drone.m, mujoco.mjtObj.mjOBJ_GEOM, geom2
                )
                
                # Получаем body ID для каждой геометрии
                geom1_body_id = self.drone.m.geom_bodyid[geom1]
                geom2_body_id = self.drone.m.geom_bodyid[geom2]
                
                # Проверяем, что контакт не с полом
                if geom1_name == "floor" or geom2_name == "floor":
                    continue
                
                # Проверяем, является ли одна из геометрий столбом (препятствием)
                is_pillar1 = geom1_name and "pillar" in geom1_name
                is_pillar2 = geom2_name and "pillar" in geom2_name
                
                # Проверяем, является ли одна из геометрий частью дрона
                # Геометрия принадлежит дрону, если её body ID совпадает с body ID дрона
                is_drone_geom1 = (geom1_body_id == drone_body_id) if drone_body_id >= 0 else False
                is_drone_geom2 = (geom2_body_id == drone_body_id) if drone_body_id >= 0 else False
                
                # Столкновение: если столб столкнулся с любой частью дрона
                if (is_pillar1 and is_drone_geom2) or (is_pillar2 and is_drone_geom1):
                    return True
        
        return False
    
    def _check_terminated(self) -> bool:
        """Проверяет условия завершения эпизода"""
        # Столкновение
        if self._check_collision():
            return True
        
        # Выход за границы
        position = self.drone.sensor.get_position()[:3]
        if position[2] < 0.1:  # Упал на землю
            return True
        if abs(position[0]) > 10 or abs(position[1]) > 10:  # Улетел слишком далеко
            return True
        
        return False
    
    def _check_truncated(self) -> bool:
        """Проверяет условия обрезания эпизода"""
        # Превышено максимальное количество шагов
        if self.step_count >= self.max_episode_steps:
            return True
        return False
    
    def _compute_reward(self) -> float:
        """
        Вычисляет награду за текущий шаг.
        
        Returns:
            Награда за шаг
        """
        reward = 0.0
        
        # Штраф за столкновение
        if self._check_collision():
            reward += self.reward_collision
            return reward
        
        # Награда за выживание (маленькая положительная награда за каждый шаг)
        reward += self.reward_survival
        
        # Награда за прогресс (движение вперед)
        if self.prev_position is not None:
            current_pos = self.drone.sensor.get_position()[:3]
            # Награда за движение вперед по оси X
            # vx отрицательна, значит движение вперед = уменьшение X
            # Поэтому прогресс = prev_x - current_x (положительное значение при движении вперед)
            progress = self.prev_position[0] - current_pos[0]
            reward += self.reward_progress * progress
        
        return reward
    
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Сбрасывает environment в начальное состояние.
        
        Args:
            seed: Семя для генератора случайных чисел
            options: Дополнительные опции
        
        Returns:
            Наблюдения и информация
        """
        super().reset(seed=seed)
        
        x0 = np.random.uniform(
            self.initial_position_range[0], self.initial_position_range[1]
        )
        y0 = np.random.uniform(
            self.initial_position_range[0], self.initial_position_range[1]
        )
        z0 = self.initial_height
        
        # Пересоздаем сцену с новыми случайными препятствиями
        self._initialize_scene_and_drone(self.initial_height, self.max_velocity, x0, y0)
        
        # Закрываем старый viewer, если он был открыт
        if self.viewer is not None:
            try:
                self.viewer.close()
            except:
                pass
            self.viewer = None
        
        # Устанавливаем начальную позицию дрона
        # Начинаем немного ниже целевой высоты для реалистичного взлета
        initial_z = max(0.2, z0 - 0.3)  # Начинаем на 0.3 м ниже целевой высоты, но не ниже 0.2 м
        self.drone.d.qpos[:3] = [x0, y0, initial_z]
        self.drone.d.qpos[3:7] = [1, 0, 0, 0]
        self.drone.d.qvel[:] = 0  # Нулевая начальная скорость
        
        # Обновляем физику для применения начального состояния
        mujoco.mj_forward(self.drone.m, self.drone.d)
        
        # Устанавливаем целевую высоту для взлета
        target_position = np.array([x0, y0, z0])
        self.drone.planner.update_target(target_position)
        
        # Сбрасываем контроллеры в автоматический режим для взлета
        self.drone.set_auto_mode()
        location = self.drone.sensor.get_position()[:3]
        # Устанавливаем setpoint на целевую высоту
        self.drone.pid_alt.setpoint = z0
        
        # Сбрасываем счетчик шагов
        self.step_count = 0
        self.prev_position = None
        
        # Фаза взлета: даем дрону время подняться до целевой высоты и стабилизироваться
        # Аналогично main.py, где дрон постепенно взлетает
        takeoff_step = 0
        for _ in range(self.takeoff_steps):
            # Обновляем внешний контроль (каждые 20 шагов или на первом шаге)
            if takeoff_step % 20 == 0 or takeoff_step == 0:
                location = self.drone.sensor.get_position()[:3]
                # Поддерживаем setpoint на целевой высоте
                self.drone.pid_alt.setpoint = z0
                self.drone.update_outer_conrol()
            
            # Обновляем внутренний контроль
            self.drone.update_inner_control()
            
            # Шаг симуляции
            mujoco.mj_step(self.drone.m, self.drone.d)
            
            takeoff_step += 1
        
        # После взлета устанавливаем финальную высоту и обновляем планировщик
        location = self.drone.sensor.get_position()[:3]
        # Ограничиваем высоту, если дрон взлетел слишком высоко
        if location[2] > z0 + 0.5:
            # Если дрон слишком высоко, устанавливаем его на правильную высоту
            self.drone.d.qpos[2] = z0
            mujoco.mj_forward(self.drone.m, self.drone.d)
        
        self.drone.pid_alt.setpoint = z0
        # Обновляем планировщик, чтобы целевая позиция была на текущей высоте
        self.drone.planner.update_target(np.array([x0, y0, z0]))
        
        # Сбрасываем счетчик шагов после взлета (шаги взлета не учитываются в эпизоде)
        self.step_count = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Выполняет один шаг симуляции.
        
        Args:
            action: Действие [vy, vz] - желаемые скорости
                   vy: скорость влево/вправо
                   vz: скорость вверх/вниз
                   vx (скорость вперед) фиксирована и всегда положительна
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Применяем действие (устанавливаем желаемые скорости)
        # action содержит только [vy, vz]
        vy = action[0]
        vz = action[1] if len(action) > 1 else 0.0
        # vx всегда фиксирована и положительна (вперед), домножаем на -1
        vx = -self.forward_velocity
        self.drone.set_velocity(vx, vy, vz)
        
        # Обновляем внешний контроль (каждые 20 шагов или на первом шаге)
        if self.step_count % 20 == 0 or self.step_count == 0:
            self.drone.update_outer_conrol()
        
        # Обновляем внутренний контроль
        self.drone.update_inner_control()
        
        # Шаг симуляции
        mujoco.mj_step(self.drone.m, self.drone.d)
        
        # Обновляем счетчик шагов
        self.step_count += 1
        
        # Получаем наблюдения
        observation = self._get_obs()
        
        # Вычисляем награду
        reward = self._compute_reward()
        
        # Проверяем условия завершения
        terminated = self._check_terminated()
        truncated = self._check_truncated()
        
        # Обновляем предыдущую позицию для вычисления прогресса
        self.prev_position = self.drone.sensor.get_position()[:3].copy()
        
        # Получаем информацию
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Рендерит environment"""
        if self.render_mode == "rgb_array":
            # Возвращаем изображение с камеры
            return self.drone.get_camera_image()
        elif self.render_mode == "human":
            # Для human режима используем mujoco.viewer
            if self.viewer is None:
                # Создаем viewer только один раз
                self.viewer = mujoco.viewer.launch_passive(
                    self.drone.m, self.drone.d
                )
            
            # Обновляем визуализацию
            if self.viewer.is_running():
                with self.viewer.lock():
                    # Можно добавить дополнительные настройки визуализации
                    pass
                self.viewer.sync()
            else:
                # Viewer был закрыт
                self.viewer = None
    
    def close(self):
        """Закрывает environment и освобождает ресурсы"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
