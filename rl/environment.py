# rl/environment.py - Gymnasium-среда дрона с препятствиями

import copy

import numpy as np
import mujoco
try:
    from mujoco import viewer as mujoco_viewer
except ImportError:
    try:
        import mujoco.viewer as mujoco_viewer
    except ImportError:
        mujoco_viewer = None
import gymnasium as gym
from gymnasium import spaces
import typing
from sim.scene import Scene
from sim.uav import Drone
from rl.map_generator import Pillar, generate_map_around_path


def _mj_quat_to_R(quat_wxyz: np.ndarray) -> np.ndarray:
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, np.asarray(quat_wxyz, dtype=np.float64))
    return mat.reshape(3, 3)


def _geom_points_body_xy_radius(m: mujoco.MjModel, gid: int) -> float:
    """
    Макс. расстояние от начала СК тела до проекции точки геометрии на XY (в СК тела),
    для выпуклой оценки — вершины ограничивающего параллелепипеда в локальных осях геома.
    """
    pos_b = np.asarray(m.geom_pos[gid], dtype=np.float64)
    R_gb = _mj_quat_to_R(np.asarray(m.geom_quat[gid], dtype=np.float64))
    sz = np.asarray(m.geom_size[gid], dtype=np.float64)
    gtype = int(m.geom_type[gid])
    locals_: typing.List[np.ndarray] = []

    if gtype == mujoco.mjtGeom.mjGEOM_BOX or gtype == mujoco.mjtGeom.mjGEOM_MESH:
        for ax in (-sz[0], sz[0]):
            for ay in (-sz[1], sz[1]):
                for az in (-sz[2], sz[2]):
                    locals_.append(np.array([ax, ay, az], dtype=np.float64))
    elif gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        for ax in (-sz[0], sz[0]):
            for ay in (-sz[1], sz[1]):
                for az in (-sz[2], sz[2]):
                    locals_.append(np.array([ax, ay, az], dtype=np.float64))
    elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        r = float(sz[0])
        for ax in (-r, r, 0.0, 0.0):
            for ay in (0.0, 0.0, -r, r):
                locals_.append(np.array([ax, ay, 0.0], dtype=np.float64))
    elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        rad, half_h = float(sz[0]), float(sz[1])
        for z in (-half_h, half_h):
            for k in range(8):
                ang = (k / 8.0) * (2.0 * np.pi)
                locals_.append(
                    np.array([rad * np.cos(ang), rad * np.sin(ang), z], dtype=np.float64)
                )
    elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        rad, half_h = float(sz[0]), float(sz[1])
        cap = half_h + rad
        for ax in (-rad, rad):
            for ay in (-rad, rad):
                for az in (-cap, cap):
                    locals_.append(np.array([ax, ay, az], dtype=np.float64))
    else:
        return 0.0

    max_xy = 0.0
    for p_l in locals_:
        p_b = R_gb @ p_l + pos_b
        max_xy = max(max_xy, float(np.hypot(float(p_b[0]), float(p_b[1]))))
    return max_xy


def drone_xy_half_span_from_mj_model(m: mujoco.MjModel, body_name: str = "x2") -> float:
    """Горизонтальный «полуразмах» дрона: max по collision-геомам тела (ограничитель сверху для XY)."""
    bid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name))
    if bid < 0:
        raise ValueError(f"В mjModel нет body {body_name!r}")
    max_r = 0.0
    for gid in range(int(m.ngeom)):
        if int(m.geom_bodyid[gid]) != bid:
            continue
        if int(m.geom_contype[gid]) == 0 and int(m.geom_conaffinity[gid]) == 0:
            continue
        max_r = max(max_r, _geom_points_body_xy_radius(m, gid))
    return float(max_r) if max_r > 1e-6 else 0.05


class DroneObstacleAvoidanceEnv(gym.Env):
    """
    Gymnasium environment для обучения БПЛА избеганию препятствий.
    
    Observation:
        - Изображение с камеры БПЛА (H, W, 3)
        - Данные акселерометра (3,) - линейное ускорение
        - Данные гироскопа (3,) - угловая скорость
    
    Action:
        - Вектор (2,): [vx, vy] в связанной СК.
          Высота — через планировщик (vz из действия не используется).
          Для обратной совместимости также принимается скаляр/shape=(1,) как vx, а vy берётся из forward_velocity.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(
        self,
        render_mode: typing.Optional[str] = None,
        max_velocity: float = 2.0,
        forward_velocity: float = 1.0,
        max_episode_steps: int = 1000,
        reward_collision: float = -150.0,
        reward_exit_failure: float = -140.0,
        reward_survival: float = -0.002,
        reward_progress: float = 0.0,
        reward_progress_max_per_step: float = 0.25,
        reward_deviation_lateral_coef: float = 0.008,
        reward_lateral_deviation_gate_low: float = 0.48,
        reward_lateral_deviation_gate_high: float = 0.88,
        reward_safe_goal_scale_min: float = 0.08,
        reward_clearance_gate_low: float = 0.12,
        reward_clearance_gate_high: float = 0.52,
        reward_barrier_coef: float = 4.0,
        reward_barrier_scale: float = 0.14,
        reward_goal_ungated_fraction: float = 0.4,
        reward_goal_distance_progress_coef: float = 6.0,
        reward_goal_reached: float = 100.0,
        reward_goal_time_coef: float = 30.0,
        reward_clearance_coef: float = 2.0,
        reward_clearance_threshold: float = 0.42,
        reward_orbit_deficit_coef: float = 2.2,
        reward_orbit_clearance_margin: float = 0.14,
        reward_action_l2_coef: float = 0.02,
        reward_action_smooth_coef: float = 0.03,
        goal_xy_tolerance: float = 0.6,
        initial_position_range: typing.Tuple[float, float] = (-2.0, 2.0),
        initial_height: float = 1.0,
        takeoff_steps: int = 500,
        num_obstacles: int = 5,  # Используется как максимальное количество (реальное определяется шумом)
        obstacle_path_length: float = 5.0,
        min_passage_width: float = 1.5,  # Минимальная ширина прохода между столбами
        show_training_target_pillar: bool = True,
        map_type: str = "random",
        single_center_radius: float = 0.45,
    ):
        """
        Инициализация environment.
        
        Args:
            render_mode: Режим рендеринга ('human' или 'rgb_array')
            max_velocity: Максимальная скорость дрона для бокового движения и высоты
            forward_velocity: Модуль крейсерской скорости вдоль +Y (тело = мир; см. step --> set_velocity)
            max_episode_steps: Максимальное количество шагов в эпизоде
            reward_collision: Штраф за столкновение (должен быть сильнее накопленного step-штрафа)
            reward_exit_failure: Терминальный штраф за out_of_bounds / out_of_corridor / fallen
            reward_survival: Награда/штраф за каждый шаг (обычно небольшой штраф по времени)
            reward_progress: Бонус за движение вперёд по +Y (0 = отключено; умножается на безопасность)
            reward_progress_max_per_step: Верхняя граница прогресса +Y за шаг
            reward_deviation_lateral_coef: Штраф за |x - ref_x|, только при большом зазоре (свободный полёт)
            reward_lateral_deviation_gate_low/high: плавное включение бокового штрафа по min_clearance
            reward_safe_goal_scale_min: нижняя граница масштаба shaping к цели при нулевом зазоре
            reward_clearance_gate_low/high: зазор, при котором shaping к цели полностью/частично включён
            reward_barrier_coef / reward_barrier_scale: эксп. барьер exp(-clearance/scale)
            reward_goal_ungated_fraction: доля shaping к цели всегда на полной силе (0–1), остальное x safety_w
            reward_goal_distance_progress_coef: Множитель за уменьшение дистанции до цели в XY
            reward_goal_reached: Бонус за достижение цели по XY (терминальная награда)
            reward_goal_time_coef: Доп. бонус за скорость достижения (доля оставшихся шагов)
            reward_clearance_coef: Штраф за близость к препятствиям ниже порога reward_clearance_threshold
            reward_clearance_threshold: Мин. желаемый зазор поверхность–поверхность (r дрона из collision-геомов x2)
            reward_orbit_deficit_coef: Множитель штрафа за «внутри» цилиндра безопасной дуги вокруг столба
            reward_orbit_clearance_margin: Доп. воздух за пределами R+r на номинальной круговой траектории
            reward_action_l2_coef: Штраф за величину управления ||a_t||^2
            reward_action_smooth_coef: Штраф за резкие изменения управления ||a_t-a_(t-1)||^2
            goal_xy_tolerance: Радиус цели в плоскости XY относительно (ref_x, ref_y)
            initial_position_range: Диапазон начальных позиций (x, y)
            initial_height: Целевая высота дрона после взлета
            takeoff_steps: Количество шагов для фазы взлета (по умолчанию 100)
            num_obstacles: Максимальное количество препятствий (столбов) в сцене (по умолчанию 5)
                          Реальное количество определяется шумом
            obstacle_path_length: Длина пути, на котором размещаются препятствия (по умолчанию 10.0)
            min_passage_width: Минимальная ширина прохода между столбами (по умолчанию 1.5)
            show_training_target_pillar: Красный маркер цели (start_x, start_y + path_length) по +Y; без коллизий.
            map_type: Тип карты: "random" (обычная генерация) или "single_center"
                (один столб ровно между стартом и целевой точкой по +Y).
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.max_velocity = max_velocity
        self.forward_velocity = forward_velocity  # Модуль; в step: vy тела = +forward_velocity (+Y мира)
        self.max_episode_steps = max_episode_steps
        self.reward_collision = reward_collision
        self.reward_exit_failure = reward_exit_failure
        self.reward_survival = reward_survival
        self.reward_progress = reward_progress
        self.reward_progress_max_per_step = reward_progress_max_per_step
        self.reward_deviation_lateral_coef = float(reward_deviation_lateral_coef)
        self.reward_lateral_deviation_gate_low = float(reward_lateral_deviation_gate_low)
        self.reward_lateral_deviation_gate_high = float(reward_lateral_deviation_gate_high)
        self.reward_safe_goal_scale_min = float(reward_safe_goal_scale_min)
        self.reward_clearance_gate_low = float(reward_clearance_gate_low)
        self.reward_clearance_gate_high = float(reward_clearance_gate_high)
        self.reward_barrier_coef = float(reward_barrier_coef)
        self.reward_barrier_scale = float(reward_barrier_scale)
        self.reward_goal_ungated_fraction = float(
            np.clip(float(reward_goal_ungated_fraction), 0.0, 1.0)
        )
        self.reward_goal_distance_progress_coef = reward_goal_distance_progress_coef
        self.reward_goal_reached = reward_goal_reached
        self.reward_goal_time_coef = reward_goal_time_coef
        self.reward_clearance_coef = reward_clearance_coef
        self.reward_clearance_threshold = reward_clearance_threshold
        self.drone_xy_half_span = 0.0  # заполняется из mjModel в _initialize_scene_and_drone
        self.reward_orbit_deficit_coef = float(reward_orbit_deficit_coef)
        self.reward_orbit_clearance_margin = float(reward_orbit_clearance_margin)
        self.reward_action_l2_coef = reward_action_l2_coef
        self.reward_action_smooth_coef = reward_action_smooth_coef
        self.goal_xy_tolerance = goal_xy_tolerance
        self.initial_position_range = initial_position_range
        self.ref_x = 0.0
        self.ref_y = 0.0
        self.ref_z = 0.0
        self.initial_height = initial_height
        self.takeoff_steps = takeoff_steps  # Количество шагов для взлета
        self.num_obstacles = num_obstacles  # Максимальное количество столбов
        self.obstacle_path_length = obstacle_path_length
        self.min_passage_width = min_passage_width  # Минимальная ширина прохода
        self.show_training_target_pillar = bool(show_training_target_pillar)
        if map_type not in ("random", "single_center"):
            raise ValueError(f"Неизвестный map_type={map_type!r}; ожидается 'random' или 'single_center'")
        self.map_type = str(map_type)
        self.single_center_radius = float(single_center_radius)
        self._random_zone_center_xy = np.zeros(2, dtype=np.float64)
        self._random_zone_radius = float(self.obstacle_path_length)

        self.scene = None
        self.drone = None
        
        # Инициализируем сцену и дрон с начальными препятствиями
        self._initialize_scene_and_drone(initial_height, max_velocity, 0.0, 0.0)
        
        # Буфер сенсоров за последние K шагов для одометрии: (K, 6) — [accel_x,y,z, gyro_x,y,z]
        self.sensor_buffer_len = 32
        self.sensor_buffer = np.zeros((self.sensor_buffer_len, 6), dtype=np.float32)
        
        # Определяем observation space (размер изображения — нативный размер камеры дрона)
        camera_height = self.drone.camera_height
        camera_width = self.drone.camera_width
        # Изображение: (H, W, 3) в диапазоне [0, 255]
        # Буфер сенсоров за K шагов (одометрия): (K, 6) — [accel, gyro] на каждом шаге
        # accelerometer, gyroscope — текущий шаг (для совместимости)
        self.observation_space = spaces.typing.Dict({
            "image": spaces.Box(
                low=0,
                high=255,
                shape=(camera_height, camera_width, 3),
                dtype=np.uint8,  # RGB (MuJoCo renderer)
            ),
            "sensor_buffer": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.sensor_buffer_len, 6),
                dtype=np.float32
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
        
        # Действие: [vx, vy] (двумерное управление скоростью в плоскости XY).
        self.action_space = spaces.Box(
            low=np.array([-max_velocity, -max_velocity], dtype=np.float32),
            high=np.array([max_velocity, max_velocity], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )
        
        # Счетчик шагов
        self.step_count = 0
        self.prev_position = None
        self.prev_goal_distance = None
        self.prev_action = None
        self.last_action = np.zeros((2,), dtype=np.float32)
        self.last_termination_reason: str | None = None
        self._last_reward_terms: typing.Dict[str, float] = {}
        
        self.viewer = None
        
    def _initialize_scene_and_drone(
        self,
        initial_height,
        max_velocity,
        start_x=0.0,
        start_y=0.0,
        pillar_factory: typing.Optional[typing.Callable[[float, float], typing.List[Pillar]]] = None,
        map_seed: typing.Optional[int] = None,
    ):
        """
        Инициализирует сцену с препятствиями и дрон
        
        Args:
            initial_height: Начальная высота дрона
            max_velocity: Максимальная скорость дрона
            start_x: Начальная X координата дрона (для размещения препятствий)
            start_y: Начальная Y координата дрона (для размещения препятствий)
            pillar_factory: Если задана, вызывается как factory(start_x, start_y) и возвращает список Pillar
                вместо стандартного generate_map_around_path.
        """
        # Создаем сцену
        self.scene = Scene()
        # Для random-карты считаем границу out по реальной зоне генерации столбов.
        self._random_zone_center_xy = np.array(
            [float(start_x), float(start_y) + 0.5 * float(self.obstacle_path_length)],
            dtype=np.float64,
        )
        self._random_zone_radius = float(self.obstacle_path_length)
        
        # Генерируем карту столбов с использованием генератора карты
        if map_seed is not None:
            noise_seed = int(map_seed)
        else:
            noise_seed = int(np.random.randint(0, 10000))
        
        if pillar_factory is not None:
            pillars = pillar_factory(float(start_x), float(start_y))
        elif self.num_obstacles <= 0:
            pillars = []
        elif self.map_type == "single_center":
            mid_y = float(start_y) + 0.5 * float(self.obstacle_path_length)
            pillars = [Pillar(x=float(start_x), y=mid_y, radius=float(self.single_center_radius), height=3.0)]
        else:
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
        
        # Добавляем столбы в сцену и сохраняем их параметры для A* и других сценариев
        self.pillars = []
        for i, pillar in enumerate(pillars):
            self.scene.add_pillar(
                x=pillar.x,
                y=pillar.y,
                radius=pillar.radius,
                height=pillar.height,
                name=f"pillar_{i+1}"
            )
            self.pillars.append((pillar.x, pillar.y, pillar.radius, pillar.height))

        if self.show_training_target_pillar:
            z_ref = float(initial_height)
            marker_h = max(2.2 * z_ref, 4.0)
            marker_r = 0.1
            marker_y = float(start_y) + float(self.obstacle_path_length)
            self.scene.add_pillar(
                x=float(start_x),
                y=marker_y,
                radius=marker_r,
                height=marker_h,
                name="training_target_marker",
                rgba=[0.95, 0.12, 0.12, 1.0],
                enable_collision=False,
            )
        
        self.drone = Drone(target=np.array((0, 0, initial_height)), scene=self.scene)
        self.drone.max_velocity = max_velocity
        self.drone_xy_half_span = drone_xy_half_span_from_mj_model(self.drone.m, body_name="x2")
    
    def _get_obs(self) -> typing.Dict[str, np.ndarray]:
        """
        Получает текущие наблюдения.
        
        Returns:
            Словарь с наблюдениями: image, sensor_buffer (одометрия за K шагов), accelerometer, gyroscope
        """
        # RGB с рендера MuJoCo — без BGR<->RGB на каждом шаге (раньше лишний cv2.cvtColor)
        camera_image = self.drone.get_camera_image_rgb()
        # Получаем данные сенсоров через методы класса Drone
        accel_data = self.drone.get_accelerometer_data()
        gyro_data = self.drone.get_gyroscope_data()
        # Обновляем буфер для одометрии: сдвиг влево, в конец — текущие [accel, gyro]
        self.sensor_buffer = np.roll(self.sensor_buffer, -1, axis=0)
        self.sensor_buffer[-1] = np.concatenate([accel_data, gyro_data])
        
        return {
            "image": camera_image,
            "sensor_buffer": self.sensor_buffer.copy(),
            "accelerometer": accel_data,
            "gyroscope": gyro_data,
        }
    
    def _get_info(self) -> typing.Dict[str, typing.Any]:
        """Возвращает дополнительную информацию для отладки"""
        position = self.drone.sensor.get_position()[:3]
        velocity = self.drone.get_current_velocity()
        dx = float(position[0]) - float(self.ref_x)
        dy = float(position[1]) - float(self.ref_y)
        
        return {
            "position": position,
            "velocity": velocity,
            "step": self.step_count,
            "distance_to_goal": float(np.sqrt(dx * dx + dy * dy)),
            "min_clearance": float(self._nearest_obstacle_clearance()),
            "drone_xy_half_span": float(self.drone_xy_half_span),
            "reward_terms": dict(self._last_reward_terms),
            "termination_reason": self.last_termination_reason,
        }

    def _goal_safety_weight(self, clearance: float) -> float:
        """w ∈ [w_min, 1]: при малом зазоре слабее бонус к цели / прогресс +Y."""
        if not np.isfinite(clearance):
            return 1.0
        w_min = float(self.reward_safe_goal_scale_min)
        lo = float(self.reward_clearance_gate_low)
        hi = float(self.reward_clearance_gate_high)
        if hi <= lo:
            hi = lo + 1e-3
        t = (float(clearance) - lo) / (hi - lo)
        t = float(np.clip(t, 0.0, 1.0))
        return w_min + (1.0 - w_min) * t

    def _lateral_deviation_weight(self, clearance: float) -> float:
        """Плавное включение штрафа за |x - ref_x|: у препятствия ≈0 (обход без доп. штрафа)."""
        if not np.isfinite(clearance):
            return 1.0
        lo = float(self.reward_lateral_deviation_gate_low)
        hi = float(self.reward_lateral_deviation_gate_high)
        if hi <= lo:
            hi = lo + 1e-3
        t = (float(clearance) - lo) / (hi - lo)
        return float(np.clip(t, 0.0, 1.0))

    def _barrier_penalty(self, clearance: float) -> float:
        if not np.isfinite(clearance):
            return 0.0
        scale = max(float(self.reward_barrier_scale), 1e-6)
        return float(self.reward_barrier_coef) * float(np.exp(-float(clearance) / scale))

    def _nearest_obstacle_clearance(self) -> float:
        """
        Минимальный зазор «поверхность препятствия — эквивалентная оболочка дрона» в XY:
        расстояние центр–центр минус R_столба и горизонтальный полуразмах дрона.
        """
        if not getattr(self, "pillars", None):
            return float("inf")
        position = self.drone.sensor.get_position()[:3]
        px, py = float(position[0]), float(position[1])
        r_d = float(self.drone_xy_half_span)
        best = float("inf")
        for ox, oy, radius, _ in self.pillars:
            dist_cc = float(np.hypot(px - float(ox), py - float(oy)))
            surf = float(dist_cc - float(radius) - r_d)
            if surf < best:
                best = surf
        return best

    def _orbit_path_deficit_penalty(self) -> float:
        """
        Учёт ширины столба: центр дрона должен оставаться вне цилиндра радиуса R + r_d + margin
        (номинальная круговая траектория обхода в плоскости XY), иначе накапливается дефицит.
        Широкий столб требует большего выноса — штраф за фиксированное «нарушение» растёт с R.
        """
        if not getattr(self, "pillars", None) or float(self.reward_orbit_deficit_coef) == 0.0:
            return 0.0
        position = self.drone.sensor.get_position()[:3]
        px, py = float(position[0]), float(position[1])
        r_d = float(self.drone_xy_half_span)
        margin = float(self.reward_orbit_clearance_margin)
        # max по столбам — сумма по всем давала гигантский штраф и «заморозку» вдали от цели
        max_deficit = 0.0
        for ox, oy, radius, _ in self.pillars:
            dist_cc = float(np.hypot(px - float(ox), py - float(oy)))
            need = float(radius) + r_d + margin
            if dist_cc < need:
                max_deficit = max(max_deficit, float(need - dist_cc))
        return float(self.reward_orbit_deficit_coef) * max_deficit
    
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
    
    def _termination_reason(self) -> str | None:
        """Возвращает причину терминального завершения или None."""
        # Столкновение
        if self._check_collision():
            return "collision"
        if self._check_goal_reached():
            return "goal_reached"
        
        # Выход за границы
        position = self.drone.sensor.get_position()[:3]
        if position[2] < 0.1:  # Упал на землю
            return "fallen"
        if self.map_type == "random":
            if self._outside_random_generation_zone(position):
                return "out_of_corridor"
        elif abs(float(position[0]) - float(self.ref_x)) > self._corridor_half_width():
            return "out_of_corridor"
        if abs(position[0]) > 10 or abs(position[1]) > 10:  # Улетел слишком далеко
            return "out_of_bounds"
        
        return None

    def _check_terminated(self) -> bool:
        """Проверяет условия завершения эпизода."""
        return self._termination_reason() is not None

    def _check_goal_reached(self) -> bool:
        """Цель достигнута, если дрон в радиусе goal_xy_tolerance от (ref_x, ref_y) в плоскости XY."""
        position = self.drone.sensor.get_position()[:3]
        dx = float(position[0]) - float(self.ref_x)
        dy = float(position[1]) - float(self.ref_y)
        return (dx * dx + dy * dy) ** 0.5 <= float(self.goal_xy_tolerance)

    def _corridor_half_width(self) -> float:
        """
        Полуширина коридора по X:
        max_obstacle_width / 2 + 0.5 = max_radius + 0.5.
        Если препятствий нет — не ограничиваем коридор.
        """
        if not getattr(self, "pillars", None):
            return float("inf")
        max_radius = max(float(p[2]) for p in self.pillars)
        return max_radius + 1.0

    def _outside_random_generation_zone(self, position: np.ndarray) -> bool:
        """Для random-карты: вне круга, в котором генерируются препятствия."""
        if self.map_type != "random":
            return False
        dx = float(position[0]) - float(self._random_zone_center_xy[0])
        dy = float(position[1]) - float(self._random_zone_center_xy[1])
        r = max(1e-6, float(self._random_zone_radius))
        return dx * dx + dy * dy > r * r
    
    def _check_truncated(self) -> bool:
        """Проверяет условия обрезания эпизода"""
        # Превышено максимальное количество шагов
        if self.step_count >= self.max_episode_steps:
            return True
        return False
    
    def _compute_reward(self, action_xy: np.ndarray, termination_reason: str | None = None) -> float:
        """
        Награда: приоритет безопасного сближения с целью; у препятствий ослабляется shaping к цели,
        усиливается барьер; боковой штраф к ref_x только при большом зазоре.
        """
        terms: typing.Dict[str, float] = {}
        reward = 0.0
        
        if termination_reason == "collision":
            reward += float(self.reward_collision)
            terms["collision"] = float(self.reward_collision)
            self._last_reward_terms = terms
            return reward
        
        surv = float(self.reward_survival)
        reward += surv
        terms["survival"] = surv
        
        current_pos = self.drone.sensor.get_position()[:3]
        x, y = float(current_pos[0]), float(current_pos[1])
        current_goal_distance = float(
            np.sqrt((x - self.ref_x) ** 2 + (y - self.ref_y) ** 2)
        )
        clearance = float(self._nearest_obstacle_clearance())
        w_goal = float(self._goal_safety_weight(clearance))
        fg = float(self.reward_goal_ungated_fraction)
        w_blend = fg + (1.0 - fg) * w_goal
        terms["safety_w"] = w_goal
        terms["goal_w_blend"] = w_blend
        terms["clearance"] = clearance if np.isfinite(clearance) else float("nan")

        # К цели: часть сигнала всегда полная (иначе агент «отползает» и растёт mean_dist_to_goal).
        goal_term = 0.0
        if self.prev_goal_distance is not None:
            goal_progress = float(self.prev_goal_distance) - current_goal_distance
            goal_term = (
                float(self.reward_goal_distance_progress_coef) * float(goal_progress) * w_blend
            )
            reward += goal_term
        terms["goal_distance"] = goal_term

        # Опциональный бонус +Y (не дублирует цель при reward_progress=0).
        prog_term = 0.0
        if float(self.reward_progress) != 0.0 and self.prev_position is not None:
            raw_progress = y - float(self.prev_position[1])
            progress = max(0.0, float(raw_progress))
            progress = min(progress, float(self.reward_progress_max_per_step))
            prog_term = float(self.reward_progress) * progress * w_blend
            reward += prog_term
        terms["progress_y"] = prog_term

        # Линейный зазор + орбита + экспоненциальный барьер.
        clear_lin = 0.0
        if np.isfinite(clearance) and clearance < float(self.reward_clearance_threshold):
            clear_lin = -float(self.reward_clearance_coef) * float(self.reward_clearance_threshold - clearance)
            reward += clear_lin
        terms["clearance_linear"] = clear_lin

        orb = -float(self._orbit_path_deficit_penalty())
        reward += orb
        terms["orbit"] = orb

        bar = -float(self._barrier_penalty(clearance))
        reward += bar
        terms["barrier"] = bar

        lat_w = float(self._lateral_deviation_weight(clearance))
        dev_x = -float(self.reward_deviation_lateral_coef) * abs(x - float(self.ref_x)) * lat_w
        reward += dev_x
        terms["dev_x"] = dev_x
        terms["lateral_w"] = lat_w

        a = action_xy.astype(np.float32)
        a_pen = -float(self.reward_action_l2_coef) * float(np.dot(a, a))
        reward += a_pen
        terms["action_l2"] = a_pen
        sm_pen = 0.0
        if self.prev_action is not None:
            da = a - self.prev_action
            sm_pen = -float(self.reward_action_smooth_coef) * float(np.dot(da, da))
            reward += sm_pen
        terms["action_smooth"] = sm_pen

        term_bonus = 0.0
        if termination_reason == "goal_reached":
            remaining_frac = max(
                0.0,
                float(self.max_episode_steps - self.step_count) / max(1.0, float(self.max_episode_steps)),
            )
            term_bonus = float(self.reward_goal_reached) + float(self.reward_goal_time_coef) * remaining_frac
            reward += term_bonus
        elif termination_reason in ("out_of_bounds", "out_of_corridor", "fallen"):
            term_bonus = float(self.reward_exit_failure)
            reward += term_bonus
        terms["terminal"] = term_bonus

        terms["total"] = float(reward)
        self._last_reward_terms = terms
        return reward
    
    def reset(
        self, seed: typing.Optional[int] = None, options: typing.Optional[typing.Dict] = None
    ) -> typing.Tuple[typing.Dict[str, np.ndarray], typing.Dict[str, typing.Any]]:
        """
        Сбрасывает environment в начальное состояние.
        
        Args:
            seed: Семя для генератора случайных чисел
            options: Дополнительные опции
        
        Returns:
            Наблюдения и информация
        """
        super().reset(seed=seed)

        lo, hi = self.initial_position_range
        x0 = float(self.np_random.uniform(lo, hi))
        y0 = float(self.np_random.uniform(lo, hi))
        z0 = self.initial_height

        pillar_factory = None
        if options is not None:
            pillar_factory = options.get("pillar_factory")

        map_seed = int(self.np_random.integers(0, 1_000_000))
        # Пересоздаем сцену с новыми случайными препятствиями — только RNG среды (не глобальный np.random)
        self._initialize_scene_and_drone(
            self.initial_height,
            self.max_velocity,
            x0,
            y0,
            pillar_factory=pillar_factory,
            map_seed=map_seed,
        )
        
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
        
        # Сбрасываем счетчик шагов и буфер сенсоров (одометрия с нуля)
        self.step_count = 0
        self.prev_position = None
        self.prev_goal_distance = None
        self.prev_action = None
        self.last_action[:] = 0.0
        self.last_termination_reason = None
        self._last_reward_terms = {}
        self.sensor_buffer.fill(0)
        
        # Фаза взлёта: много подряд mj_step — заметная стоимость каждого reset (см. takeoff_steps в __init__)
        for _ in range(self.takeoff_steps):
            self.drone.pid_alt.setpoint = z0
            self.drone.update_outer_conrol()
            self.drone.update_inner_control()
            mujoco.mj_step(self.drone.m, self.drone.d)
        
        # После взлета устанавливаем финальную высоту и обновляем планировщик
        location = self.drone.sensor.get_position()[:3]
        # Ограничиваем высоту, если дрон взлетел слишком высоко
        if location[2] > z0 + 0.5:
            # Если дрон слишком высоко, устанавливаем его на правильную высоту
            self.drone.d.qpos[2] = z0
            mujoco.mj_forward(self.drone.m, self.drone.d)
            location = self.drone.sensor.get_position()[:3]

        # Удержание высоты при set_velocity(vz=None) идёт через planner.get_alt_setpoint;
        # совпадаем z цели с фактической высотой после взлёта (иначе мелкий разрыв с z0 даёт
        # просадку при движении с наклоном).
        z_hold = float(location[2])
        self.drone.pid_alt.setpoint = z_hold
        self.drone.planner.update_target(np.array([x0, y0, z_hold], dtype=np.float64))

        # После взлёта: нулевая скорость и нулевой рыскание (оси тела = миру).
        nv = int(self.drone.m.nv)
        n = min(6, nv)
        self.drone.d.qvel[:n] = 0.0
        self.drone.d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.drone.m, self.drone.d)
        
        # Сбрасываем счетчик шагов после взлета (шаги взлета не учитываются в эпизоде)
        self.step_count = 0
        self.ref_x = float(x0)
        self.ref_y = float(y0) + float(self.obstacle_path_length)
        self.ref_z = z_hold
        self.prev_goal_distance = float(
            np.sqrt((float(location[0]) - self.ref_x) ** 2 + (float(location[1]) - self.ref_y) ** 2)
        )
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self, action: np.ndarray
    ) -> typing.Tuple[typing.Dict[str, np.ndarray], float, bool, bool, typing.Dict[str, typing.Any]]:
        """
        Выполняет один шаг симуляции.
        
        Args:
            action: [vx, vy] в системе тела. Для совместимости допускается shape=(1,): тогда vy=forward_velocity.
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Полёт к точке (set_target): среда не задаёт скорости — только планировщик дрона.
        if not self.drone.target_navigation_active:
            a = np.asarray(action, dtype=np.float32).reshape(-1)
            if a.size == 0:
                raise ValueError("action не должен быть пустым")
            vx = float(a[0])
            vy = float(a[1]) if a.size >= 2 else float(self.forward_velocity)
            vx = float(np.clip(vx, float(self.action_space.low[0]), float(self.action_space.high[0])))
            vy = float(np.clip(vy, float(self.action_space.low[1]), float(self.action_space.high[1])))
            action_xy = np.array([vx, vy], dtype=np.float32)
            self.last_action = action_xy
            self.drone.set_velocity(vx, vy, vz=None)
        else:
            action_xy = self.last_action.copy()

        # Внешний контур (скорость XY + уставка высоты) нужен каждый шаг; редкий вызов ломал удержание высоты и слежение за vx/vy.
        self.drone.update_outer_conrol()
        
        # Обновляем внутренний контроль
        self.drone.update_inner_control()
        
        # Шаг симуляции
        mujoco.mj_step(self.drone.m, self.drone.d)
        
        # Обновляем счетчик шагов
        self.step_count += 1
        
        # Получаем наблюдения
        observation = self._get_obs()
        
        # Проверяем условия завершения
        termination_reason = self._termination_reason()
        terminated = termination_reason is not None
        truncated = self._check_truncated()
        if not terminated and truncated:
            termination_reason = "timeout"
        self.last_termination_reason = termination_reason

        # Вычисляем награду
        reward = self._compute_reward(action_xy=action_xy, termination_reason=termination_reason)
        
        # Обновляем предыдущую позицию для вычисления прогресса
        self.prev_position = self.drone.sensor.get_position()[:3].copy()
        cur = self.prev_position
        self.prev_goal_distance = float(
            np.sqrt((float(cur[0]) - self.ref_x) ** 2 + (float(cur[1]) - self.ref_y) ** 2)
        )
        self.prev_action = action_xy.copy()
        
        # Получаем информацию
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Рендерит environment"""
        if self.render_mode == "rgb_array":
            return self.drone.get_camera_image_rgb()
        elif self.render_mode == "human":
            if mujoco_viewer is None:
                raise RuntimeError(
                    "MuJoCo viewer недоступен. Используйте `mjpython` или запуск без GUI (--no-render)."
                )
            if self.viewer is None:
                self.viewer = mujoco_viewer.launch_passive(
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

    def _snapshot_drone_control(self) -> dict:
        d = self.drone
        return {
            "manual_control": d.manual_control,
            "manual_velocities": d.manual_velocities.copy(),
            "target_navigation_active": d.target_navigation_active,
            "_target_xy_tolerance": d._target_xy_tolerance,
            "_target_z_tolerance": d._target_z_tolerance,
            "pid_alt": copy.deepcopy(d.pid_alt),
            "pid_roll": copy.deepcopy(d.pid_roll),
            "pid_pitch": copy.deepcopy(d.pid_pitch),
            "pid_yaw": copy.deepcopy(d.pid_yaw),
            "pid_v_x": copy.deepcopy(d.pid_v_x),
            "pid_v_y": copy.deepcopy(d.pid_v_y),
            "planner": copy.deepcopy(d.planner),
        }

    def _restore_drone_control(self, template: dict) -> None:
        # deepcopy: в снимке лежат «эталонные» ПИД/планировщик; при restore нельзя
        # присваивать те же объекты дрону — иначе env.step() в роллаутах MCTS меняет
        # интеграторы прямо внутри snap["drone_ctrl"], и следующий restore уже не корень.
        s = copy.deepcopy(template)
        d = self.drone
        d.manual_control = s["manual_control"]
        d.manual_velocities[:] = s["manual_velocities"]
        d.target_navigation_active = s["target_navigation_active"]
        d._target_xy_tolerance = s["_target_xy_tolerance"]
        d._target_z_tolerance = s["_target_z_tolerance"]
        d.pid_alt = s["pid_alt"]
        d.pid_roll = s["pid_roll"]
        d.pid_pitch = s["pid_pitch"]
        d.pid_yaw = s["pid_yaw"]
        d.pid_v_x = s["pid_v_x"]
        d.pid_v_y = s["pid_v_y"]
        d.planner = s["planner"]

    def snapshot_for_branching(self) -> dict:
        """Снимок для MCTS: полное интеграционное состояние MuJoCo, буфер сенсоров, ПИДы, ref_*."""
        dd = self.drone.d
        m = self.drone.m
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        n = mujoco.mj_stateSize(m, spec)
        mj_state = np.empty(n, dtype=np.float64)
        mujoco.mj_getState(m, dd, mj_state, spec)
        return {
            "mj_state": mj_state,
            "mj_state_spec": int(spec),
            "step_count": self.step_count,
            "prev_position": None if self.prev_position is None else np.copy(self.prev_position),
            "prev_goal_distance": None if self.prev_goal_distance is None else float(self.prev_goal_distance),
            "prev_action": None if self.prev_action is None else np.copy(self.prev_action),
            "last_action": np.copy(self.last_action),
            "last_termination_reason": self.last_termination_reason,
            "sensor_buffer": np.copy(self.sensor_buffer),
            "ref_x": float(self.ref_x),
            "ref_y": float(self.ref_y),
            "ref_z": float(self.ref_z),
            "drone_ctrl": self._snapshot_drone_control(),
        }

    def restore_from_branching_snapshot(self, snap: dict) -> None:
        dd = self.drone.d
        m = self.drone.m
        if "mj_state" in snap and "mj_state_spec" in snap:
            mujoco.mj_setState(m, dd, snap["mj_state"], snap["mj_state_spec"])
            mujoco.mj_forward(m, dd)
        else:
            # Совместимость со старыми снимками (до mj_getState)
            dd.qpos[:] = snap["qpos"]
            dd.qvel[:] = snap["qvel"]
            dd.act[:] = snap["act"]
            dd.ctrl[:] = snap["ctrl"]
            dd.time = snap["time"]
            mujoco.mj_forward(m, dd)
        self.step_count = snap["step_count"]
        self.prev_position = (
            None if snap["prev_position"] is None else np.copy(snap["prev_position"])
        )
        self.prev_goal_distance = None if snap.get("prev_goal_distance") is None else float(snap["prev_goal_distance"])
        self.prev_action = None if snap.get("prev_action") is None else np.asarray(snap["prev_action"], dtype=np.float32).copy()
        if "last_action" in snap:
            self.last_action = np.asarray(snap["last_action"], dtype=np.float32).copy()
        else:
            self.last_action = np.zeros((2,), dtype=np.float32)
        self.last_termination_reason = snap.get("last_termination_reason")
        self.sensor_buffer[:] = snap["sensor_buffer"]
        self.ref_x = float(snap.get("ref_x", self.ref_x))
        self.ref_y = float(snap["ref_y"])
        self.ref_z = float(snap["ref_z"])
        self._restore_drone_control(snap["drone_ctrl"])
