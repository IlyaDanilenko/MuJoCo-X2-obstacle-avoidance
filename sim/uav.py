# sim/uav.py - код взаимодействия с моделью БПЛА, функции управления взяты из оригинального репозитория модели

import mujoco
import cv2
import numpy as np
from simple_pid import PID
from .controllers import DummyPlanner, DummySensor

class Drone:
  def __init__(self, target=np.array((0,0,0)), scene=None):
    """
    Инициализация БПЛА
    
    Args:
        target: Целевая позиция БПЛА
        scene: Объект Scene, в котором находится дрон. Если None, загружается только модель коптера
    """
    self.m = scene.m
    self.d = scene.d

    self.planner = DummyPlanner(
        target=target, sim_dt=float(self.m.opt.timestep)
    )
    self.sensor = DummySensor(self.d)

    # instantiate controllers

    # inner control to stabalize inflight dynamics
    # sample_time=None + dt=opt.timestep при вызове: иначе при шаге симуляции <0.01 с
    # simple_pid по умолчанию «замораживает» выход между интеграциями MuJoCo.
    _st = None
    self.pid_alt = PID(9.5, 0.85, 1.2, setpoint=0, sample_time=_st)
    self.pid_roll = PID(2.6785,0.56871, 1.2508, setpoint=0, output_limits = (-1,1), sample_time=_st)
    self.pid_pitch = PID(2.6785,0.56871, 1.2508, setpoint=0, output_limits = (-1,1), sample_time=_st)
    self.pid_yaw = PID(0.54, 0, 5.358333, setpoint=0, output_limits=(-3, 3), sample_time=_st)

    # outer control loops: выход — задание тангажа/крена (рад); ±0.1 мало для слежения за v_xy
    _tilt_lim = 0.42
    self.pid_v_x = PID(0.1, 0.003, 0.02, setpoint=0, output_limits=(-_tilt_lim, _tilt_lim), sample_time=_st)
    self.pid_v_y = PID(0.1, 0.003, 0.02, setpoint=0, output_limits=(-_tilt_lim, _tilt_lim), sample_time=_st)
    
    # Управление скоростями
    self.manual_control = False  # Режим ручного управления
    self.manual_velocities = np.array([0.0, 0.0, 0.0])  # Желаемые скорости в ручном режиме
    # Лимит уставки крена/тангажа при set_velocity (ручной режим), рад (~20°)
    self._manual_max_tilt_rad = float(np.deg2rad(40.0))
    self.max_velocity = 2.0  # Максимальная скорость
    self.velocity_step = 0.1  # Шаг изменения скорости при управлении

    # Полёт к точке (планировщик); при True среда не перезаписывает скорости через set_velocity
    self.target_navigation_active = False
    self._target_xy_tolerance = 0.07
    self._target_z_tolerance = 0.2
    
    # Камера дрона
    self.camera_id = "drone_camera"
    self.camera_width = 640
    self.camera_height = 480
    self.renderer = mujoco.Renderer(self.m, height=self.camera_height, width=self.camera_width)
    
    # Получаем индексы сенсоров
    self.gyro_id = mujoco.mj_name2id(
        self.m, mujoco.mjtObj.mjOBJ_SENSOR, "body_gyro"
    )
    self.accel_id = mujoco.mj_name2id(
        self.m, mujoco.mjtObj.mjOBJ_SENSOR, "body_linacc"
    )
    
    # Проверяем, что сенсоры найдены
    if self.gyro_id < 0:
        raise ValueError("Сенсор 'body_gyro' не найден в модели!")
    if self.accel_id < 0:
      raise ValueError("Сенсор 'body_linacc' не найден в модели!")

    self._x2_body_id = int(mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "x2"))
    if self._x2_body_id < 0:
      raise ValueError("Body 'x2' не найден в модели")

  def get_world_linear_velocity(self):
    """
    Линейная скорость центра x2 в мировой СК (vx, vy, vz) в м/с.
    Нельзя брать d.qvel[:3]: у free joint первые три компонента — угловая скорость.
    """
    vel6 = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        self.m,
        self.d,
        mujoco.mjtObj.mjOBJ_BODY,
        self._x2_body_id,
        vel6,
        0,
    )
    return vel6[3:6].astype(np.float32)

  def _x2_rotmat_world(self) -> np.ndarray:
    """Матрица 3x3 ориентации тела x2: локальные оси в мировой СК (как у mjData.xmat)."""
    bid = int(self._x2_body_id)
    xm = np.asarray(self.d.xmat, dtype=np.float64)
    if xm.ndim == 2 and xm.shape[1] == 9:
      return xm[bid].reshape(3, 3)
    if xm.ndim == 3 and xm.shape[1:] == (3, 3):
      return np.array(xm[bid], dtype=np.float64, copy=False)
    if xm.ndim == 2 and xm.shape[0] == self.m.nbody * 3 and xm.shape[1] == 3:
      return np.array(xm[bid * 3 : bid * 3 + 3, :], dtype=np.float64, copy=True)
    flat = xm.ravel()
    if flat.size == self.m.nbody * 9:
      return flat[bid * 9 : (bid + 1) * 9].reshape(3, 3)
    raise RuntimeError(f"Неожиданная форма mjData.xmat: {xm.shape}")

  def _body_rpy_world(self) -> tuple[float, float, float]:
    """
    Roll, pitch, yaw (рад) из матрицы ориентации; ZYX от мира к телу.
    Не использовать qpos[3:7] — там кватернион, не углы Эйлера.
    """
    R = self._x2_rotmat_world()
    sinp = -float(np.clip(R[2, 0], -1.0, 1.0))
    pitch = float(np.arcsin(sinp))
    cosp = float(np.cos(pitch))
    if abs(cosp) < 1e-8:
      roll = 0.0
      yaw = float(np.arctan2(-R[0, 1], R[1, 1]))
    else:
      roll = float(np.arctan2(R[2, 1], R[2, 2]))
      yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw

  def set_velocity(self, vx, vy, vz=None):
    """Уставки скорости в связанной СК тела x2; при совпадении осей с миром (после взлёта в RL):
    vx>0 — вправо, vx<0 — влево; vy>0 — вперёд по +Y, vy<0 — назад. Компоненты переводятся
    в мир через матрицу ориентации и идут на ПИД по мировым vx, vy.
    
    Args:
        vx (float): м/с вдоль оси X тела
        vy (float): м/с вдоль оси Y тела
        vz (float, optional): для высоты (планировщик при None).
    """
    self.manual_control = True
    self.manual_velocities[0] = np.clip(vx, -self.max_velocity, self.max_velocity)
    self.manual_velocities[1] = np.clip(vy, -self.max_velocity, self.max_velocity)
    if vz is not None:
      self.manual_velocities[2] = np.clip(vz, -self.max_velocity, self.max_velocity)
    else:
      self.manual_velocities[2] = 0.0

  def set_auto_mode(self):
    """Переключает в автоматический режим (использует планировщик)"""
    self.manual_control = False

  def set_target(
    self,
    target,
    xy_tolerance=0.07,
    z_tolerance=0.2,
  ):
    """
    Лететь к заданной точке в мировых координатах (планировщик DummyPlanner).

    Args:
        target: (2,) — X, Y, Z берётся с текущей высоты; или (3,) — полная позиция.
        xy_tolerance: порог по горизонтали (м) для point_reached.
        z_tolerance: порог по высоте (м) для point_reached.
    """
    loc = self.sensor.get_position()[:3]
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    if t.size == 2:
      goal = np.array([t[0], t[1], float(loc[2])], dtype=np.float64)
    elif t.size == 3:
      goal = t.astype(np.float64).copy()
    else:
      raise ValueError("target: ожидается 2 или 3 компонента")

    self.planner.update_target(goal)
    self.planner.vel_limit = float(self.max_velocity)
    lim = float(self.max_velocity)
    self.planner.pid_x.output_limits = (-lim, lim)
    self.planner.pid_y.output_limits = (-lim, lim)

    self._target_xy_tolerance = float(xy_tolerance)
    self._target_z_tolerance = float(z_tolerance)
    self.manual_control = False
    self.target_navigation_active = True

  def clear_target_navigation(self):
    """Выключает полёт к точке; дальше снова действуют set_velocity из среды."""
    self.target_navigation_active = False
    self.manual_control = True
    self.manual_velocities[:] = 0.0

  @property
  def point_reached(self):
    """True, если нет активной навигации к точке или позиция в пределах допуска от цели."""
    if not self.target_navigation_active:
      return True
    loc = self.sensor.get_position()[:3]
    g = np.asarray(self.planner.target, dtype=np.float64).reshape(3)
    dxy = float(np.linalg.norm(loc[:2] - g[:2]))
    dz = float(abs(loc[2] - g[2]))
    return dxy <= self._target_xy_tolerance and dz <= self._target_z_tolerance

  def get_current_velocity(self):
    """Линейная скорость в мировой СК (vx, vy, vz), м/с."""
    return self.get_world_linear_velocity()

  def get_camera_image_rgb(self):
    """Кадр с камеры RGB uint8 (H,W,3) — для нейросетей без лишнего cv2.cvtColor."""
    self.renderer.update_scene(self.d, camera=self.camera_id)
    return np.asarray(self.renderer.render(), dtype=np.uint8)

  def get_camera_rgb_and_depth(self):
    """
    Один проход offscreen: глубина (метры от камеры) + RGB, как у кадра наблюдения.
    После вызова рендерер снова в режиме RGB (disable_depth_rendering).
    """
    self.renderer.enable_depth_rendering()
    self.renderer.update_scene(self.d, camera=self.camera_id)
    depth = np.asarray(self.renderer.render(), dtype=np.float32)
    self.renderer.disable_depth_rendering()
    self.renderer.update_scene(self.d, camera=self.camera_id)
    rgb = np.asarray(self.renderer.render(), dtype=np.uint8)
    return rgb, depth

  def get_camera_image(self):
    """BGR для OpenCV (imshow/imwrite); внутри один вызов рендера + конвертация."""
    rgb = self.get_camera_image_rgb()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

  def get_gyroscope_data(self):
    """
    Получает данные гироскопа (угловая скорость).
    
    Returns:
        np.array: Данные гироскопа [wx, wy, wz] в формате float32
    """
    gyro_start_idx = self.m.sensor_adr[self.gyro_id]
    gyro_dim = self.m.sensor_dim[self.gyro_id]
    gyro_data = self.d.sensordata[
        gyro_start_idx : gyro_start_idx + gyro_dim
    ].copy()
    return gyro_data.astype(np.float32)

  def get_accelerometer_data(self):
    """
    Получает данные акселерометра (линейное ускорение).
    
    Returns:
        np.array: Данные акселерометра [ax, ay, az] в формате float32
    """
    accel_start_idx = self.m.sensor_adr[self.accel_id]
    accel_dim = self.m.sensor_dim[self.accel_id]
    accel_data = self.d.sensordata[
        accel_start_idx : accel_start_idx + accel_dim
    ].copy()
    return accel_data.astype(np.float32)

  def update_outer_conrol(self):
    """Updates outer control loop for trajectory planning"""
    v_lin = self.get_world_linear_velocity()
    location = self.sensor.get_position()[:3]

    if self.manual_control:
      # Уставки заданы в связанной СК тела --> перевод в мир для ПИД по мировым Vx, Vy
      vx_b = float(self.manual_velocities[0])
      vy_b = float(self.manual_velocities[1])
      R = self._x2_rotmat_world()
      v_horiz_w = R @ np.array([vx_b, vy_b, 0.0], dtype=np.float64)
      self.pid_v_x.setpoint = float(v_horiz_w[0])
      self.pid_v_y.setpoint = float(v_horiz_w[1])

      # Если задана скорость по Z, используем её, иначе используем планировщик для высоты
      if self.manual_velocities[2] != 0:
        current_alt = location[2]
        self.pid_alt.setpoint = current_alt + self.manual_velocities[2] * 0.1
      else:
        self.pid_alt.setpoint = self.planner.get_alt_setpoint(location)
    else:
      # Автоматический режим - используем планировщик (оригинальный)
      # Compute velocites to target
      velocites = self.planner(loc=location)
      
      # In this example the altitude is directly controlled by a PID
      self.pid_alt.setpoint = self.planner.get_alt_setpoint(location)
      self.pid_v_x.setpoint = velocites[0]
      self.pid_v_y.setpoint = velocites[1]

    dt = float(self.m.opt.timestep)
    # Мировые vx, vy --> крен/тангаж (X2): vx --> тангаж, vy --> крен (инверсия vy по крену).
    angle_roll = -self.pid_v_y(float(v_lin[1]), dt=dt)
    angle_pitch = self.pid_v_x(float(v_lin[0]), dt=dt)

    if self.manual_control:
      lim = self._manual_max_tilt_rad
      angle_roll = float(np.clip(angle_roll, -lim, lim))
      angle_pitch = float(np.clip(angle_pitch, -lim, lim))

    self.pid_pitch.setpoint = angle_pitch
    self.pid_roll.setpoint = angle_roll

  def update_inner_control(self):
    """Upates inner control loop and sets actuators to stabilize flight
    dynamics"""
    alt = float(self.sensor.get_position()[2])
    roll, pitch, yaw = self._body_rpy_world()

    # При крене/тангаже вертикальная составляющая тяги падает ~∝ cos(наклон) — без масштаба дрон оседает.
    # R[:,2] — ось +Z тела в мировой СК; R[2,2] — её проекция на мировой «вверх».
    R = self._x2_rotmat_world()
    up_frac = float(np.clip(abs(R[2, 2]), 0.14, 1.0))
    tilt_thrust_gain = min(1.0 / up_frac, 2.58)

    hover_trim = 3.38
    v_lin = self.get_world_linear_velocity()
    vz = float(v_lin[2])
    vz_damp = 0.1
    dt = float(self.m.opt.timestep)
    cmd_thrust = (self.pid_alt(alt, dt=dt) + hover_trim) * tilt_thrust_gain - vz_damp * vz
    cmd_roll = - self.pid_roll(roll, dt=dt)
    cmd_pitch = self.pid_pitch(pitch, dt=dt)
    cmd_yaw = - self.pid_yaw(yaw, dt=dt)

    #transfer to motor control
    out = self.compute_motor_control(cmd_thrust, cmd_roll, cmd_pitch, cmd_yaw)
    self.d.ctrl[:4] = out

  #  as the drone is underactuated we set
  def compute_motor_control(self, thrust, roll, pitch, yaw):
    motor_control = [
      thrust + roll + pitch - yaw,
      thrust - roll + pitch + yaw,
      thrust - roll -  pitch - yaw,
      thrust + roll - pitch + yaw
    ]
    return motor_control