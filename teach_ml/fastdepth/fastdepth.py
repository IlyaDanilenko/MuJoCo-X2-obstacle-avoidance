# teach_ml/fastdepth/fastdepth.py - сеть FastDepth и обёртка инференса

import importlib.util
import sys
import urllib.request
from pathlib import Path
import typing

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- пути и загрузка весов -------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PRETRAINED_DIR = _PROJECT_ROOT / "pretrained_model"
DEFAULT_WEIGHTS_NAME = "FastDepth_L1GN_Best.pth"
DEFAULT_WEIGHTS_URL = (
    "https://raw.githubusercontent.com/Hagaik92/FastDepth/main/Weights/FastDepth_L1GN_Best.pth"
)
FASTDEPTH_BACKBONE = "mobilenet"
_MIN_VALID_BYTES = 100_000  # чекпойнт FastDepth заметно больше; меньше — скорее HTML/LFS pointer


def _download_weights(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = DEFAULT_WEIGHTS_URL
    print(f"FastDepth: скачивание весов\n  {url}\n  -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "MuJoCo-X2-obstacle-avoidance/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = resp.read()
    if len(data) < _MIN_VALID_BYTES:
        raise ValueError(f"слишком малый ответ ({len(data)} B), возможно не .pth")
    dest.write_bytes(data)
    print(f"FastDepth: сохранено {len(data) / (1024 * 1024):.1f} MiB")


def resolve_weights_path(weights_path: typing.Optional[typing.Union[str, Path]] = None) -> Path:
    """
    Если weights_path задан и файл есть — возвращает его.
    Иначе путь по умолчанию: pretrained_model/FastDepth_L1GN_Best.pth;
    если файла нет — скачивает с GitHub в этот путь (или в родителя указанного пути).
    """
    if weights_path:
        p = Path(weights_path).expanduser().resolve()
    else:
        PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
        p = PRETRAINED_DIR / DEFAULT_WEIGHTS_NAME
    if p.is_file():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    _download_weights(p)
    return p


def _ensure_nyudepthv2() -> None:
    """
    Чекпойнты FastDepth ссылаются на модуль nyudepthv2 (NYUDataset и т.д.).
    Подключаем вендорную копию teach_ml/nyudepthv2.py под именем nyudepthv2.
    Источник: https://github.com/Hagaik92/FastDepth/blob/main/nyudepthv2.py
    """
    if "nyudepthv2" in sys.modules:
        return
    vendored = Path(__file__).resolve().parent / "nyudepthv2.py"
    spec = importlib.util.spec_from_file_location("nyudepthv2", vendored)
    if spec is None or spec.loader is None:
        raise ImportError(f"Не найден вендорный nyudepthv2: {vendored}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nyudepthv2"] = mod
    spec.loader.exec_module(mod)


def _load_fastdepth_checkpoint(path: Path) -> dict:
    """Сначала безопасная загрузка тензоров; иначе полный pickle с nyudepthv2."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        _ensure_nyudepthv2()
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Ожидался dict в чекпойнте, получено {type(ckpt)}")
    return ckpt


# --- сеть (адаптировано из FastDepth/models.py, только MobileNet FastDepth) --


def ConvBlock(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def ConvReLU6Block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


def DWConvBlock(in_channels, out_channels, kernel_size, stride, padding=None):
    if padding is None:
        padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, groups=in_channels
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def DWConvReLU6Block(in_channels, out_channels, kernel_size, stride, padding=None):
    if padding is None:
        padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, groups=in_channels
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


class MobileNet_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_layer1 = ConvReLU6Block(3, 32, 3, 2, 1)
        self.enc_layer2 = nn.Sequential(DWConvReLU6Block(32, 32, 3, 1, 1), ConvReLU6Block(32, 64, 1, 1, 0))
        self.enc_layer3 = nn.Sequential(DWConvReLU6Block(64, 64, 3, 2, 1), ConvReLU6Block(64, 128, 1, 1, 0))
        self.enc_layer4 = nn.Sequential(DWConvReLU6Block(128, 128, 3, 1, 1), ConvReLU6Block(128, 128, 1, 1, 0))
        self.enc_layer5 = nn.Sequential(DWConvReLU6Block(128, 128, 3, 2, 1), ConvReLU6Block(128, 256, 1, 1, 0))
        self.enc_layer6 = nn.Sequential(DWConvReLU6Block(256, 256, 3, 1, 1), ConvReLU6Block(256, 256, 1, 1, 0))
        self.enc_layer7 = nn.Sequential(DWConvReLU6Block(256, 256, 3, 2, 1), ConvReLU6Block(256, 512, 1, 1, 0))
        self.enc_layer8 = nn.Sequential(DWConvReLU6Block(512, 512, 3, 1, 1), ConvReLU6Block(512, 512, 1, 1, 0))
        self.enc_layer9 = nn.Sequential(DWConvReLU6Block(512, 512, 3, 1, 1), ConvReLU6Block(512, 512, 1, 1, 0))
        self.enc_layer10 = nn.Sequential(DWConvReLU6Block(512, 512, 3, 1, 1), ConvReLU6Block(512, 512, 1, 1, 0))
        self.enc_layer11 = nn.Sequential(DWConvReLU6Block(512, 512, 3, 1, 1), ConvReLU6Block(512, 512, 1, 1, 0))
        self.enc_layer12 = nn.Sequential(DWConvReLU6Block(512, 512, 3, 1, 1), ConvReLU6Block(512, 512, 1, 1, 0))
        self.enc_layer13 = nn.Sequential(DWConvReLU6Block(512, 512, 3, 2, 1), ConvReLU6Block(512, 1024, 1, 1, 0))
        self.enc_layer14 = nn.Sequential(DWConvReLU6Block(1024, 1024, 3, 1, 1), ConvReLU6Block(1024, 1024, 1, 1, 0))
        self.enc_layer15 = nn.AvgPool2d(7)
        self.enc_output = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.enc_layer1(x)
        x = self.enc_layer2(x)
        x = self.enc_layer3(x)
        x = self.enc_layer4(x)
        x = self.enc_layer5(x)
        x = self.enc_layer6(x)
        x = self.enc_layer7(x)
        x = self.enc_layer8(x)
        x = self.enc_layer9(x)
        x = self.enc_layer10(x)
        x = self.enc_layer11(x)
        x = self.enc_layer12(x)
        x = self.enc_layer13(x)
        x = self.enc_layer14(x)
        x = self.enc_layer15(x)
        return self.enc_output(x)


class NNConv5_Decoder(nn.Module):
    def __init__(self, kernel_size, depthwise=True):
        super().__init__()
        if depthwise:
            self.conv1 = nn.Sequential(DWConvBlock(1024, 1024, kernel_size, 1), ConvBlock(1024, 512, 1, 1, 0))
            self.conv2 = nn.Sequential(DWConvBlock(512, 512, kernel_size, 1), ConvBlock(512, 256, 1, 1, 0))
            self.conv3 = nn.Sequential(DWConvBlock(256, 256, kernel_size, 1), ConvBlock(256, 128, 1, 1, 0))
            self.conv4 = nn.Sequential(DWConvBlock(128, 128, kernel_size, 1), ConvBlock(128, 64, 1, 1, 0))
            self.conv5 = nn.Sequential(DWConvBlock(64, 64, kernel_size, 1), ConvBlock(64, 32, 1, 1, 0))
        else:
            self.conv1 = ConvBlock(1024, 512, kernel_size, 1, (kernel_size - 1) // 2)
            self.conv2 = ConvBlock(512, 256, kernel_size, 1, (kernel_size - 1) // 2)
            self.conv3 = ConvBlock(256, 128, kernel_size, 1, (kernel_size - 1) // 2)
            self.conv4 = ConvBlock(128, 64, kernel_size, 1, (kernel_size - 1) // 2)
            self.conv5 = ConvBlock(64, 32, kernel_size, 1, (kernel_size - 1) // 2)
        self.output = ConvBlock(32, 1, 1, 1, 0)

    def forward(self, x):
        x = F.interpolate(self.conv1(x), scale_factor=2, mode="nearest")
        x = F.interpolate(self.conv2(x), scale_factor=2, mode="nearest")
        x = F.interpolate(self.conv3(x), scale_factor=2, mode="nearest")
        x = F.interpolate(self.conv4(x), scale_factor=2, mode="nearest")
        x = F.interpolate(self.conv5(x), scale_factor=2, mode="nearest")
        return self.output(x)


class FastDepth(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.encoder = MobileNet_Encoder()
        self.decoder = NNConv5_Decoder(kernel_size)

    def forward(self, x):
        x = self.encoder.enc_layer1(x)
        x = self.encoder.enc_layer2(x)
        layer1 = x
        x = self.encoder.enc_layer3(x)
        x = self.encoder.enc_layer4(x)
        layer2 = x
        x = self.encoder.enc_layer5(x)
        x = self.encoder.enc_layer6(x)
        layer3 = x
        x = self.encoder.enc_layer7(x)
        x = self.encoder.enc_layer8(x)
        x = self.encoder.enc_layer9(x)
        x = self.encoder.enc_layer10(x)
        x = self.encoder.enc_layer11(x)
        x = self.encoder.enc_layer12(x)
        x = self.encoder.enc_layer13(x)
        x = self.encoder.enc_layer14(x)
        x = F.interpolate(self.decoder.conv1(x), scale_factor=2, mode="nearest")
        x = F.interpolate(self.decoder.conv2(x), scale_factor=2, mode="nearest")
        x = x + layer3
        x = F.interpolate(self.decoder.conv3(x), scale_factor=2, mode="nearest") + layer2
        x = F.interpolate(self.decoder.conv4(x), scale_factor=2, mode="nearest") + layer1
        x = F.interpolate(self.decoder.conv5(x), scale_factor=2, mode="nearest")
        return self.decoder.output(x)


# --- обёртка ---------------------------------------------------------------


class FastDepthWrapper(nn.Module):
    """
    Чекпойнт с ключом model_state_dict (как в оригинальном FastDepth).
    weights_path=None --> pretrained_model/FastDepth_L1GN_Best.pth (скачивание при отсутствии).
    """

    iheight = 480
    iwidth = 640

    def __init__(self, weights_path: typing.Optional[typing.Union[str, Path]] = None):
        super().__init__()
        path = resolve_weights_path(weights_path)
        self.weights_path = path
        ckpt = _load_fastdepth_checkpoint(path)
        if "model_state_dict" not in ckpt:
            raise KeyError("В чекпойнте нет ключа 'model_state_dict' (ожидается сохранение FastDepth).")

        self.depth_net = FastDepth()
        self.depth_net.load_state_dict(ckpt["model_state_dict"], strict=True)
        for p in self.depth_net.parameters():
            p.requires_grad = False
        self.depth_net.eval()

    def preprocess_rgb_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        if x.max() > 1.0:
            x = x / 255.0
        x = torch.clamp(x, 0.0, 1.0)

        h1 = int(250 * self.iheight / self.iheight)
        w1 = int(250 * self.iwidth / self.iheight)
        x = F.interpolate(x, size=(h1, w1), mode="bilinear", align_corners=False)

        ch, cw = 228, 304
        _, _, H, W = x.shape
        top = max(0, (H - ch) // 2)
        left = max(0, (W - cw) // 2)
        x = x[:, :, top : top + ch, left : left + cw]

        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return x

    def forward(self, x224: torch.Tensor) -> torch.Tensor:
        return self.depth_net(x224)

    def rgb_tensor_to_depth_normalized(self, image: torch.Tensor) -> torch.Tensor:
        x = self.preprocess_rgb_tensor(image)
        with torch.inference_mode():
            depth = self.depth_net(x)
            d_min = depth.amin(dim=(2, 3), keepdim=True)
            d_max = depth.amax(dim=(2, 3), keepdim=True)
            return (depth - d_min) / (d_max - d_min + 1e-6)

    def numpy_hwc_to_depth_normalized_tensor(
        self,
        rgb_hwc: np.ndarray,
        device: typing.Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Кадр HWC uint8/float --> нормализованная глубина [1,1,224,224] на device.
        Без .cpu(): для EnvEncoder / политики после одного H2D кадра вся цепочка остаётся на GPU.
        """
        dev = device or next(self.depth_net.parameters()).device
        if rgb_hwc.dtype != np.uint8:
            im = np.clip(rgb_hwc, 0, 255).astype(np.float32) / 255.0
        else:
            im = rgb_hwc.astype(np.float32) / 255.0
        t = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).to(dev)
        return self.rgb_tensor_to_depth_normalized(t)

    def rgb_tensor_to_policy_input(self, image: torch.Tensor) -> torch.Tensor:
        d = self.rgb_tensor_to_depth_normalized(image)
        return d.expand(-1, 3, -1, -1)

    def numpy_rgb_to_depth_uint8(
        self,
        rgb_hwc: np.ndarray,
        device: typing.Optional[torch.device] = None,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        dev = device or next(self.depth_net.parameters()).device
        if rgb_hwc.dtype != np.uint8:
            im = np.clip(rgb_hwc, 0, 255).astype(np.float32) / 255.0
        else:
            im = rgb_hwc.astype(np.float32) / 255.0
        t = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).to(dev)
        x = self.preprocess_rgb_tensor(t)
        with torch.inference_mode():
            depth_t = self.depth_net(x).squeeze(0).float()
        depth = depth_t.cpu().numpy()
        d_min, d_max = depth.min(), depth.max()
        d_n = (depth - d_min) / (d_max - d_min + 1e-6)
        u8 = (d_n * 255.0).astype(np.uint8)
        return u8, depth
