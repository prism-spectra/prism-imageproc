# %% Imports
from __future__ import annotations
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
from typing import ClassVar, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload
from xarray import Dataset, DataArray, concat, MergeError, load_dataset
from astropy.units import Quantity
import astropy.units as u
from skimage.transform import AffineTransform
from scipy.ndimage import map_coordinates
from natsort import natsorted
from dataclasses import dataclass, field
from numpy import arange, asarray, interp, sqrt, nan, ndarray
from numpy.typing import NDArray
from serde_dataclass import json_config, toml_config, JsonDataclass, TomlDataclass
import astropy_xarray as _

from .utils import QuantityEncoder, QUANTITY_DECODER

# %% Definitions

ScaleType = Union[float, Tuple[float, float]]
TranslationType = Tuple[float, float]
PixelSizeType = Tuple[float, float]
PaddingMode = Literal['constant', 'edge', 'symmetric', 'reflect', 'wrap']

# Map skimage-only mode names to scipy.ndimage equivalents.
_SKIMAGE_TO_SCIPY_MODE: Dict[str, str] = {
    'edge': 'nearest',
    'symmetric': 'mirror',
}


@dataclass
@json_config(ser=QuantityEncoder, de=QUANTITY_DECODER)
@toml_config(de=QUANTITY_DECODER)
class TransformMatrix(JsonDataclass, TomlDataclass):
    """Reusable affine transform state and composition helper."""

    matrix: ndarray = field(
        default_factory=lambda: asarray([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float),
        metadata={
            'description': '3x3 affine transformation matrix in homogeneous coordinates.',
            'typecheck': lambda x, _: isinstance(x, (list, ndarray)) and asarray(x).shape == (3, 3),
        }
    )

    def __post_init__(self) -> None:
        self.matrix = asarray(self.matrix, dtype=float)
        if self.matrix.shape != (3, 3):
            raise ValueError('matrix must have shape (3, 3)')

    @classmethod
    def from_matrix(
            cls,
            matrix: NDArray,
    ) -> TransformMatrix:
        return cls(matrix=asarray(matrix, dtype=float))

    def append(self, affine: AffineTransform) -> None:
        self.matrix = affine.params @ self.matrix

    def reset(self) -> None:
        self.matrix = asarray([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

    def affine(self) -> AffineTransform:
        return AffineTransform(matrix=self.matrix.copy())

    def effective_scale(self) -> Tuple[float, float]:
        a = float(self.matrix[0, 0])
        b = float(self.matrix[0, 1])
        d = float(self.matrix[1, 0])
        e = float(self.matrix[1, 1])
        sx = float(sqrt(a * a + d * d))
        sy = float(sqrt(b * b + e * e))
        return abs(sx), abs(sy)


@dataclass
@json_config(ser=QuantityEncoder, de=QUANTITY_DECODER)
@toml_config(de=QUANTITY_DECODER)
class MosaicImageMapper(JsonDataclass):
    """Map an image onto mosaic coordinates using a provided affine matrix.

    This helper is intentionally lightweight compared with ``MosaicImageTransform``:
    it requires only source image axes, mosaic bounds, and a transformation matrix.
    """

    source_x: ndarray
    source_y: ndarray
    target_x: ndarray
    target_y: ndarray
    pixel_size: PixelSizeType
    bounds_x: Tuple[float, float]
    bounds_y: Tuple[float, float]
    transform: TransformMatrix = field(
        default_factory=TransformMatrix)
    _source_x0: float = field(init=False)
    _source_y0: float = field(init=False)
    _inv_dx: float = field(init=False)
    _inv_dy: float = field(init=False)
    _use_linear_x: bool = field(init=False)
    _use_linear_y: bool = field(init=False)

    def __post_init__(self) -> None:
        self.source_x = asarray(self.source_x, dtype=float)
        self.source_y = asarray(self.source_y, dtype=float)
        self.target_x = asarray(self.target_x, dtype=float)
        self.target_y = asarray(self.target_y, dtype=float)
        if self.source_x.ndim != 1 or self.source_y.ndim != 1:
            raise ValueError('source_x and source_y must be 1D arrays')
        if self.target_x.ndim != 1 or self.target_y.ndim != 1:
            raise ValueError('target_x and target_y must be 1D arrays')
        if self.target_x.size == 0 or self.target_y.size == 0:
            raise ValueError('target_x and target_y must not be empty')
        if self.pixel_size[0] <= 0 or self.pixel_size[1] <= 0:
            raise ValueError('pixel_size must be positive')
        if not isinstance(self.transform, TransformMatrix):
            self.transform = TransformMatrix.from_matrix(
                asarray(self.transform, dtype=float))

        # Match MosaicImageTransform coordinate-index behavior for consistency.
        self._source_x0 = float(self.source_x[0])
        self._source_y0 = float(self.source_y[0])
        self._inv_dx = 0.0
        self._inv_dy = 0.0
        self._use_linear_x = False
        self._use_linear_y = False

        if self.source_x.size >= 2:
            dx = float(self.source_x[1] - self.source_x[0])
            if dx != 0.0:
                xdiff = asarray(
                    self.source_x[1:] - self.source_x[:-1], dtype=float)
                xtol = max(1e-12, 1e-9 * abs(dx))
                self._use_linear_x = bool((abs(xdiff - dx) <= xtol).all())
                if self._use_linear_x:
                    self._inv_dx = 1.0 / dx

        if self.source_y.size >= 2:
            dy = float(self.source_y[1] - self.source_y[0])
            if dy != 0.0:
                ydiff = asarray(
                    self.source_y[1:] - self.source_y[:-1], dtype=float)
                ytol = max(1e-12, 1e-9 * abs(dy))
                self._use_linear_y = bool((abs(ydiff - dy) <= ytol).all())
                if self._use_linear_y:
                    self._inv_dy = 1.0 / dy

    def map_to_mosaic(
        self,
        image: NDArray,
        order: int = 1,
        cval: float = nan,
        mode: str = 'constant',
    ) -> Tuple[DataArray, PixelSizeType]:
        """Render a 2D image onto the finalized full-resolution mosaic grid."""
        image_data = asarray(image)
        if image_data.ndim != 2:
            raise ValueError(f'image must be a 2D array, got {image_data.ndim}D')
        if self.source_x.size != image_data.shape[1]:
            raise ValueError(f'source_x size {self.source_x.size} must match image width {image_data.shape[1]}')
        if self.source_y.size != image_data.shape[0]:
            raise ValueError(f'source_y size {self.source_y.size} must match image height {image_data.shape[0]}')

        x_target = self.target_x
        y_target = self.target_y
        px, py = float(self.pixel_size[0]), float(self.pixel_size[1])

        # Broadcasting avoids allocating two full meshgrid arrays.
        x_row = x_target[None, :]  # shape (1, nx)
        y_col = y_target[:, None]  # shape (ny, 1)
        inv = self.transform.affine().inverse.params
        src_x = inv[0, 0] * x_row + inv[0, 1] * y_col + inv[0, 2]  # (ny, nx)
        src_y = inv[1, 0] * x_row + inv[1, 1] * y_col + inv[1, 2]  # (ny, nx)

        if self._use_linear_x:
            col = self._coord_to_index_linear(
                src_x, self._source_x0, self._inv_dx, self.source_x.size)
        else:
            col = self._coord_to_index(src_x, self.source_x)

        if self._use_linear_y:
            row = self._coord_to_index_linear(
                src_y, self._source_y0, self._inv_dy, self.source_y.size)
        else:
            row = self._coord_to_index(src_y, self.source_y)

        # Translate skimage-only mode names to scipy equivalents.
        scipy_mode = _SKIMAGE_TO_SCIPY_MODE.get(mode, mode)
        warped = map_coordinates(
            image_data.astype(float, copy=False),
            [row, col],
            order=order,
            cval=cval,
            mode=scipy_mode,
            prefilter=order > 1,
        )
        out = DataArray(
            warped,
            dims=('y', 'x'),
            coords={
                'x': ('x', Quantity(x_target, u.mm), {'units': u.mm, 'description': 'Mosaic X coordinate'}),
                'y': ('y', Quantity(y_target, u.mm), {'units': u.mm, 'description': 'Mosaic Y coordinate'}),
            },
        ).astropy.quantify()
        out.attrs['pixel_scale_x_mm_per_px'] = float(px)
        out.attrs['pixel_scale_y_mm_per_px'] = float(py)
        out.attrs['bounds_x_mm'] = self.bounds_x
        out.attrs['bounds_y_mm'] = self.bounds_y
        return out, (float(px), float(py))

    @staticmethod
    def _coord_to_index(coord: NDArray, axis_values: NDArray) -> NDArray:
        """Map physical coordinates to floating pixel indices by linear interpolation.

        The source axis may be ascending or descending.
        Coordinates outside axis range map to ``-1`` and are handled by ``warp``
        according to the selected boundary ``mode`` and ``cval``.
        """
        idx = arange(axis_values.size, dtype=float)
        axis = asarray(axis_values, dtype=float)
        if axis.size < 2:
            return interp(coord, axis, idx, left=-1.0, right=-1.0)
        if axis[0] > axis[-1]:
            axis = axis[::-1]
            idx = idx[::-1]
        return interp(coord, axis, idx, left=-1.0, right=-1.0)

    @staticmethod
    def _coord_to_index_linear(
            coord: NDArray,
            axis0: float,
            inv_step: float,
            size: int,
    ) -> NDArray:
        """Map physical coordinates to floating indices for a uniform axis.

        Out-of-bounds coordinates are left as-is; ``map_coordinates`` applies
        ``cval`` for any index outside ``[0, size-1]`` when ``mode='constant'``.
        """
        return (coord - axis0) * inv_step


class MosaicImageStraightener:
    """Straighten images mapped onto the mosaic grid using curve maps associated with window names.
    """

    def __init__(
            self,
            imaps: Dict[str, List[Dataset]],
            mapper: MosaicImageMapper,
    ):
        self._mapper = mapper
        self._imaps = imaps
        self._windows = list(imaps.keys())

    @classmethod
    def load(
        cls,
        archive_path: Path,
    ) -> MosaicImageStraightener:
        """Load a MosaicImageStraightener from a binary bundle containing a MosaicImageMapper and associated curve maps.

        Args:
            archive_path (Path): The path to the archive file containing the mosaic image straightener data.

        Raises:
            ValueError: If the archive file does not exist or if required data is missing.
            ValueError: If the mosaic image mapper is not found in the archive.

        Returns:
            MosaicImageStraightener: The loaded mosaic image straightener.
        """
        if not archive_path.exists():
            raise ValueError(f'Archive {archive_path} does not exist')
        mapper = None
        with tarfile.open(archive_path, "r:xz") as tar:
            with TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                tmpdir.mkdir(exist_ok=True, parents=True)
                tar.extractall(path=tmpdir)
                imaps = {}
                for dsdir in tmpdir.iterdir():
                    if not dsdir.is_dir():
                        # Load the mapper
                        if dsdir.suffix == '.json' and '_mapper' in dsdir.stem:
                            mapper = MosaicImageMapper.from_json(
                                dsdir.read_text())
                    else:
                        # Load the curve maps
                        win_name = dsdir.name
                        datasets = []
                        for dsfile in natsorted(dsdir.iterdir()):
                            if not dsfile.is_file() or dsfile.suffix != '.nc':
                                continue
                            ds = load_dataset(dsfile)
                            datasets.append(ds)
                        imaps[win_name] = datasets
                if mapper is None:
                    raise ValueError('MosaicImageMapper not found in archive')
                return cls(imaps, mapper)

    @property
    def windows(self) -> List[str]:
        """Return the list of available window names for straightening.

        Returns:
            List[str]: The list of available window names for straightening.
        """
        return self._windows

    def load_image(
        self,
        image: NDArray,
        order: int = 1,
        cval: float = nan,
        mode: PaddingMode = 'constant',
    ) -> MosaicMappedImage:
        """Load an image onto the mosaic grid using the mapper, preparing it for straightening.

        Args:
            image (NDArray): The input image to be mapped onto the mosaic grid. Must be a 2D array.
            order (int, optional): The interpolation order to use. Defaults to 1 (bilinear). Available options are 0 (nearest), 1 (bilinear), 2 (biquadratic), 3 (bicubic), 4 (biquartic), and 5 (biquintic).
            cval (float, optional): The constant value to use for padding. Defaults to nan.
            mode (PaddingMode, optional): The padding mode to use. Defaults to 'constant'.

        Returns:
            MosaicMappedImage: The image mapped onto the mosaic grid.
        """
        da, px = self._mapper.map_to_mosaic(
            image, order, cval, mode)
        mapped = MosaicMappedImage(
            image=da,
            pixel_size=px
        )
        mapped._imaps = self._imaps
        mapped._windows = self._windows
        return mapped


@dataclass
class MosaicMappedImage:
    """An image mapped onto the mosaic grid, ready for straightening using associated curve maps."""
    image: DataArray
    pixel_size: PixelSizeType
    _imaps: Dict[str, List[Dataset]] = field(init=False, repr=False)
    _windows: List[str] = field(init=False, repr=False)

    @property
    def windows(self) -> List[str]:
        """Return the list of available window names for straightening.

        Returns:
            List[str]: The list of available window names for straightening.
        """
        return self._windows

    @overload
    def straighten_image(
        self,
        window: str,
        *,
        inplace: bool = ...
    ) -> DataArray: ...

    @overload
    def straighten_image(
        self,
        window: List[str],
        *,
        inplace: bool = ...
    ) -> Dict[str, DataArray]: ...

    @overload
    def straighten_image(
        self,
        window: None = ...,
        *,
        inplace: bool = ...
    ) -> Dict[str, DataArray]: ...

    def straighten_image(
        self,
        window: Optional[str] | List[str] = None,
        *,
        inplace: bool = True
    ) -> DataArray | Dict[str, DataArray]:
        """Straighten the given image using the curve maps associated with the specified window name(s).

        Args:
            window (Optional[str | Sequence[str]]): The name(s) of the window(s) to use for straightening. If None, all available windows will be used. If a string is provided, the corresponding window will be used. If a list of strings is provided, each specified window will be used and the results will be returned in a dictionary keyed by window name. Defaults to None.
            inplace (bool, optional): If True, the input image will be modified in place. Defaults to True.

        Raises:
            ValueError: If the straightener is not properly initialized or if window is invalid.

        Returns:
            DataArray | Dict[str, DataArray]: _description_
        """
        image = self.image
        if self._imaps is None:
            raise ValueError('Must setup first')
        if window is None:
            window = self.windows
        if isinstance(window, str):
            ret: List = []
            for ds in self._imaps.get(window, []):
                # Trick to select the same location across different datasets
                imageset = Dataset(
                    data_vars={
                        'image': (('y', 'x'), image.values),
                        'loc': (('y', 'x'), ds['loc'].data, ds['loc'].attrs),
                    },
                    coords={
                        'y': (('y',), image['y'].values, {'units': u.mm, 'description': 'Mosaic coordinate, increasing from the bottom.'}),
                        'x': (('x',), image['x'].values, {'units': u.mm, 'description': 'Mosaic coordinate, increasing from the left.'}),
                    },
                ).astropy.quantify()
                xran = ds.attrs['xran']
                yran = ds.attrs['yran']
                xran = (Quantity(xran[0], u.mm), Quantity(xran[1], u.mm))
                yran = (Quantity(yran[0], u.mm), Quantity(yran[1], u.mm))
                xform = ds['xform'].data
                res = ds['resolution']
                data = imageset.where(imageset['loc'], drop=True)['image']
                data = data.sel(y=slice(*yran), x=slice(*xran))
                if inplace:
                    try:
                        data /= res.data
                    except (MergeError, ValueError):
                        data = data.data / res.data
                else:
                    data = data / res.data
                data = map_coordinates(
                    data.values[:, :], xform, cval=nan, order=1, prefilter=False)
                out = DataArray(data*10, coords={
                    'y': (
                        ('y',),
                        ds['wly'].data,  # type: ignore
                        {
                            'units': 'mm',
                            'description': 'Height in the mosaic coordinate, increasing from the bottom.'
                        }
                    ),
                    'wavelength': (
                        'wavelength',
                        ds['wavelength'].data / 10.0,  # type: ignore
                        {
                            'units': 'nm',
                            'description': 'Wavelength in nanometers',
                        }
                    ),
                })
                ret.append(out)
            out: DataArray = concat(ret, dim='y', join='outer')  # type: ignore
            out = out.sortby('y')
            out = out.sortby('wavelength')
            if image.attrs.get('units') is not None:
                out.attrs['units'] = image.attrs['units'] + ' / nm'
            else:
                out.attrs['units'] = '1 / nm'
            return out
        elif isinstance(window, Sequence):
            if len(window) == 0:
                window = self.windows
            return {
                name: self.straighten_image(name, inplace=inplace)
                for name in window
            }
        else:
            raise ValueError(
                'win_name must be a string or an iterable of strings')
