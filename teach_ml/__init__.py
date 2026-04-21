# teach_ml/__init__.py - реэкспорт FastDepth

from .fastdepth import FASTDEPTH_BACKBONE, FastDepthWrapper, resolve_weights_path

__all__ = ["FastDepthWrapper", "FASTDEPTH_BACKBONE", "resolve_weights_path"]
