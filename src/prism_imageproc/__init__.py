import importlib.metadata as metadata

try:
    __version__ = metadata.version(__package__ or 'prism_imageproc')
except metadata.PackageNotFoundError:
    # Fallback version if package metadata is not found. 
    # This should be updated if the version is changed in pyproject.toml.
    __version__ = '0.0.2'

from .straighten_image import (
    MosaicImageMapper,
    MosaicImageStraightener,
    MosaicMappedImage,
    ScaleType,
    TranslationType,
    PixelSizeType,
    TransformMatrix,
)
from . import utils

__all__ = [
    "MosaicImageMapper",
    "MosaicImageStraightener",
    "MosaicMappedImage",
    "ScaleType",
    "TranslationType",
    "PixelSizeType",
    "TransformMatrix",
    "utils",
    "__version__",
]