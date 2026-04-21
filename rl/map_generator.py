# rl/map_generator.py - генератор препятствий и карты

import numpy as np
import typing


def simple_noise(x: float, y: float, seed: int = 0) -> float:
    """
    Простая функция шума на основе синусоидальных функций.
    Создает плавное значение шума в диапазоне [-1, 1].
    
    Args:
        x: X координата
        y: Y координата
        seed: Семя для генерации шума
    
    Returns:
        Значение шума в диапазоне [-1, 1]
    """
    # Используем локальный генератор случайных чисел, чтобы не влиять на глобальное состояние
    rng = np.random.RandomState(seed)
    freq1 = rng.uniform(0.1, 0.5)
    freq2 = rng.uniform(0.3, 0.7)
    freq3 = rng.uniform(0.5, 1.0)
    
    # Комбинируем несколько частот для более сложного паттерна
    noise_value = (
        np.sin(x * freq1 + y * freq2) * 0.5 +
        np.cos(x * freq2 - y * freq1) * 0.3 +
        np.sin(x * freq3 + y * freq3) * 0.2
    )
    
    return np.clip(noise_value, -1.0, 1.0)


class Pillar:
    """Класс для представления столба"""
    def __init__(self, x: float, y: float, radius: float, height: float):
        self.x = x
        self.y = y
        self.radius = radius
        self.height = height
    
    def distance_to(self, other: 'Pillar') -> float:
        """Вычисляет расстояние между центрами двух столбов"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def passage_width(self, other: 'Pillar') -> float:
        """Вычисляет ширину прохода между двумя столбами"""
        distance = self.distance_to(other)
        return distance - self.radius - other.radius

def generate_map_around_path(
    start_x: float = 0.0,
    start_y: float = 0.0,
    path_length: float = 10.0,
    min_pillars: int = 3,
    max_pillars: int = 15,
    min_radius: float = 0.3,
    max_radius: float = 0.8,
    min_height: float = 1.0,
    max_height: float = 6.0,
    min_passage_width: float = 1.5,
    seed: typing.Optional[int] = None,
) -> typing.List[Pillar]:
    """
    Генерирует карту столбов вокруг пути движения дрона.
    Оси мира и тела совпадают: вперёд — +Y, вправо/влево — ±X. Центр области — середина
    отрезка от старта (start_x, start_y) до цели (start_x, start_y + path_length).
    
    Args:
        start_x: Начальная X координата дрона
        start_y: Начальная Y координата дрона
        path_length: Длина пути движения до цели по +Y
        min_pillars: Минимальное количество столбов
        max_pillars: Максимальное количество столбов
        min_radius: Минимальный радиус столба
        max_radius: Максимальный радиус столба
        min_height: Минимальная высота столба
        max_height: Максимальная высота столба
        min_passage_width: Минимальная ширина прохода между столбами
        seed: Семя для генерации
    
    Returns:
        Список столбов Pillar
    """
    center_x = start_x
    center_y = start_y + path_length / 2.0
    
    # Радиус области равен длине пути
    area_radius = path_length
    
    if seed is None:
        seed = np.random.randint(0, 10000)
    
    rng = np.random.RandomState(seed)
    
    # Определяем количество столбов на основе шума
    # Используем шум для более естественного распределения
    noise_count = simple_noise(area_radius * 0.1, area_radius * 0.2, seed=seed)
    # Преобразуем шум [-1, 1] в количество столбов [min_pillars, max_pillars]
    num_pillars = int(min_pillars + (noise_count + 1.0) / 2.0 * (max_pillars - min_pillars))
    num_pillars = max(min_pillars, min(max_pillars, num_pillars))
    
    pillars: typing.List[Pillar] = []
    
    for i in range(num_pillars):
        attempts = 0
        placed = False
        
        while attempts < 1000 and not placed:
            attempts += 1
            
            # Генерируем позицию в круговой области
            # Используем равномерное распределение по радиусу и углу
            angle = rng.uniform(0, 2 * np.pi)
            # Распределение по радиусу: более плотное в центре
            r_factor = np.sqrt(rng.uniform(0, 1))  # Квадратный корень для более равномерного распределения
            radius_pos = r_factor * area_radius
            
            x = center_x + radius_pos * np.cos(angle)
            y = center_y + radius_pos * np.sin(angle)
            
            # Используем шум для генерации радиуса и высоты
            noise_radius = simple_noise(x * 0.3, y * 0.3, seed=seed + 1000 + i)
            noise_height = simple_noise(x * 0.2, y * 0.2, seed=seed + 2000 + i)
            
            # Преобразуем шум в радиус и высоту
            radius = min_radius + (noise_radius + 1.0) / 2.0 * (max_radius - min_radius)
            height = min_height + (noise_height + 1.0) / 2.0 * (max_height - min_height)
            
            # Проверяем, что столб не выходит за границы области
            if radius_pos + radius > area_radius:
                continue
            
            # Создаем кандидата на столб
            candidate = Pillar(x, y, radius, height)
            
            # Проверяем проходы со всеми уже размещенными столбами
            valid = True
            for existing_pillar in pillars:
                passage = candidate.passage_width(existing_pillar)
                if passage < min_passage_width:
                    valid = False
                    break
            
            if valid:
                pillars.append(candidate)
                placed = True
        
        if not placed:
            # Если не удалось разместить столб, пропускаем его
            pass
    
    return pillars


def _circle_intersects_axis_aligned_rect(
    px: float, py: float, radius: float,
    xmin: float, xmax: float, ymin: float, ymax: float,
) -> bool:
    """True, если круг (px, py, radius) пересекает или касается осесимметричного прямоугольника."""
    cx = float(np.clip(px, xmin, xmax))
    cy = float(np.clip(py, ymin, ymax))
    return (px - cx) ** 2 + (py - cy) ** 2 <= radius * radius


def generate_map_outside_clear_rectangle(
    clear_xmin: float,
    clear_xmax: float,
    clear_ymin: float,
    clear_ymax: float,
    sample_xmin: float,
    sample_xmax: float,
    sample_ymin: float,
    sample_ymax: float,
    min_pillars: int = 3,
    max_pillars: int = 15,
    min_radius: float = 0.3,
    max_radius: float = 0.8,
    min_height: float = 1.0,
    max_height: float = 6.0,
    min_passage_width: float = 1.5,
    seed: typing.Optional[int] = None,
) -> typing.List[Pillar]:
    """
    Столбы по тому же принципу, что и generate_map_around_path: число — через simple_noise,
    радиус/высота — через simple_noise от координат, проверка проходов между столбами.

    Центры и радиусы подбираются так, чтобы круг столба не пересекал заданный «чистый»
    прямоугольник [clear_xmin, clear_xmax] x [clear_ymin, clear_ymax] (траектория и запас).

    Позиции центров равномерно в прямоугольнике выборки
    [sample_xmin, sample_xmax] x [sample_ymin, sample_ymax]; кандидаты вне области или
    пересекающие clear-rect отбрасываются.
    """
    if seed is None:
        seed = int(np.random.randint(0, 10000))
    rng = np.random.RandomState(seed)

    span_x = sample_xmax - sample_xmin
    span_y = sample_ymax - sample_ymin
    span = max(span_x, span_y, 1e-6)

    noise_count = simple_noise(span * 0.1, span * 0.2, seed=seed)
    num_pillars = int(min_pillars + (noise_count + 1.0) / 2.0 * (max_pillars - min_pillars))
    num_pillars = max(min_pillars, min(max_pillars, num_pillars))

    pillars: typing.List[Pillar] = []

    for i in range(num_pillars):
        attempts = 0
        placed = False
        while attempts < 2000 and not placed:
            attempts += 1
            x = rng.uniform(sample_xmin, sample_xmax)
            y = rng.uniform(sample_ymin, sample_ymax)

            noise_radius = simple_noise(x * 0.3, y * 0.3, seed=seed + 1000 + i)
            noise_height = simple_noise(x * 0.2, y * 0.2, seed=seed + 2000 + i)
            radius = min_radius + (noise_radius + 1.0) / 2.0 * (max_radius - min_radius)
            height = min_height + (noise_height + 1.0) / 2.0 * (max_height - min_height)

            if _circle_intersects_axis_aligned_rect(
                x, y, radius, clear_xmin, clear_xmax, clear_ymin, clear_ymax
            ):
                continue

            candidate = Pillar(x, y, radius, height)
            valid = True
            for existing in pillars:
                if candidate.passage_width(existing) < min_passage_width:
                    valid = False
                    break
            if valid:
                pillars.append(candidate)
                placed = True

    return pillars


def clear_rectangle_for_square_path(
    p0x: float,
    p0y: float,
    square_size: float = 1.0,
    margin_zone_half_extent: float = 0.6,
) -> typing.Tuple[float, float, float, float]:
    """
    Квадратная траектория 1x1 м как в imu-test: углы (p0x - square_size, p0y) --> ... --> (p0x, p0y).
    Центр квадрата: (p0x - square_size/2, p0y + square_size/2).
    Возвращает границы зоны без столбов 2*margin_zone_half_extent по X и Y (по умолчанию 1.2x1.2 м).
    """
    cx = p0x - 0.5 * square_size
    cy = p0y + 0.5 * square_size
    h = margin_zone_half_extent
    return (cx - h, cx + h, cy - h, cy + h)

