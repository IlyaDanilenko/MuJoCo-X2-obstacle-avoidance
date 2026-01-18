import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DronePolicyNetwork(nn.Module):
    """
    Политическая сеть для управления дроном.
    
    Входы:
        - Изображение с камеры (64x64x3)
        - Данные акселерометра (3,)
        - Данные гироскопа (3,)
    
    Выход:
        - Действие [vy, vz] - скорости бокового смещения и высоты
    """
    
    def __init__(self, image_size=(64, 64), action_dim=2):
        super(DronePolicyNetwork, self).__init__()
        
        self.image_size = image_size
        self.action_dim = action_dim
        
        # CNN для обработки изображения
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Вычисляем размер после сверточных слоев
        # 64x64 -> 15x15 -> 6x6 -> 4x4
        conv_output_size = 64 * 4 * 4
        
        # Размер для сенсорных данных (акселерометр + гироскоп)
        sensor_dim = 3 + 3  # 6
        
        # Полносвязные слои
        self.fc1 = nn.Linear(conv_output_size + sensor_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        
        # Выходной слой для действий (mean и std для нормального распределения)
        self.action_mean = nn.Linear(128, action_dim)
        self.action_std = nn.Linear(128, action_dim)
        
        # Инициализация весов
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Инициализация весов сети"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, image, accelerometer, gyroscope):
        """
        Прямой проход сети.
        
        Args:
            image: Изображение [batch_size, 3, H, W]
            accelerometer: Данные акселерометра [batch_size, 3]
            gyroscope: Данные гироскопа [batch_size, 3]
        
        Returns:
            action_mean: Среднее значение действия [batch_size, action_dim]
            action_std: Стандартное отклонение действия [batch_size, action_dim]
        """
        batch_size = image.size(0)
        
        # Обработка изображения через CNN
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, -1)  # Flatten
        
        # Объединяем изображение и сенсорные данные
        sensor_data = torch.cat([accelerometer, gyroscope], dim=1)
        x = torch.cat([x, sensor_data], dim=1)
        
        # Полносвязные слои
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # Выходные значения
        action_mean = torch.tanh(self.action_mean(x))  # Ограничиваем [-1, 1]
        action_std = F.softplus(self.action_std(x)) + 0.01  # Положительное значение с минимумом
        
        return action_mean, action_std
    
    def get_action(self, image, accelerometer, gyroscope, deterministic=False):
        """
        Получает действие от сети.
        
        Args:
            image: Изображение [1, 3, H, W] или [H, W, 3]
            accelerometer: Данные акселерометра [1, 3] или [3,]
            gyroscope: Данные гироскопа [1, 3] или [3,]
            deterministic: Если True, возвращает среднее значение без случайности
        
        Returns:
            action: Действие [action_dim]
        """
        self.eval()
        with torch.no_grad():
            # Подготавливаем входные данные
            # Конвертируем numpy массивы в torch тензоры
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float()
            if isinstance(accelerometer, np.ndarray):
                accelerometer = torch.from_numpy(accelerometer).float()
            if isinstance(gyroscope, np.ndarray):
                gyroscope = torch.from_numpy(gyroscope).float()
            
            # Подготавливаем размерности
            if len(image.shape) == 3:  # [H, W, 3]
                image = image.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            if len(accelerometer.shape) == 1:  # [3,]
                accelerometer = accelerometer.unsqueeze(0)  # [1, 3]
            if len(gyroscope.shape) == 1:  # [3,]
                gyroscope = gyroscope.unsqueeze(0)  # [1, 3]
            
            # Нормализуем изображение [0, 255] -> [0, 1]
            if image.max() > 1.0:
                image = image / 255.0
            
            # Перемещаем на устройство
            device = next(self.parameters()).device
            image = image.to(device)
            accelerometer = accelerometer.to(device)
            gyroscope = gyroscope.to(device)
            
            # Прямой проход
            action_mean, action_std = self.forward(image, accelerometer, gyroscope)
            
            if deterministic:
                action = action_mean.squeeze(0)
            else:
                # Сэмплируем из нормального распределения
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample().squeeze(0)
            
            return action.cpu().numpy()


class DroneValueNetwork(nn.Module):
    """
    Сеть для оценки значения состояния (value function).
    Используется в алгоритмах обучения с подкреплением.
    """
    
    def __init__(self, image_size=(64, 64)):
        super(DroneValueNetwork, self).__init__()
        
        self.image_size = image_size
        
        # CNN для обработки изображения (та же архитектура, что и в policy)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        conv_output_size = 64 * 4 * 4
        sensor_dim = 3 + 3  # акселерометр + гироскоп
        
        # Полносвязные слои
        self.fc1 = nn.Linear(conv_output_size + sensor_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        
        # Выходной слой для значения состояния
        self.value = nn.Linear(128, 1)
        
        # Инициализация весов
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Инициализация весов сети"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, image, accelerometer, gyroscope):
        """
        Прямой проход сети.
        
        Args:
            image: Изображение [batch_size, 3, H, W]
            accelerometer: Данные акселерометра [batch_size, 3]
            gyroscope: Данные гироскопа [batch_size, 3]
        
        Returns:
            value: Оценка значения состояния [batch_size, 1]
        """
        batch_size = image.size(0)
        
        # Обработка изображения через CNN
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, -1)  # Flatten
        
        # Объединяем изображение и сенсорные данные
        sensor_data = torch.cat([accelerometer, gyroscope], dim=1)
        x = torch.cat([x, sensor_data], dim=1)
        
        # Полносвязные слои
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # Выходное значение
        value = self.value(x)
        
        return value
