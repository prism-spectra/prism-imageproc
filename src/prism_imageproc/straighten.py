# %% Imports
from __future__ import annotations
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Sequence, overload
from xarray import Dataset, DataArray, concat, MergeError, load_dataset
from astropy.units import Quantity
import astropy.units as u
from scipy.ndimage import map_coordinates
from natsort import natsorted
from dataclasses import dataclass, field
from numpy import nan
from numpy.typing import NDArray
import astropy_xarray as _

from .internals import MosaicImageMapper, PaddingMode, PixelSizeType, TransformMatrix

# %% Definitions


class ImageStraightener:
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
    ) -> ImageStraightener:
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
            with TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
                tmpdir = Path(tmpdir)
                tmpdir.mkdir(exist_ok=True, parents=True)
                tar.extractall(path=tmpdir)
                imaps = {}
                for dsdir in natsorted(tmpdir.iterdir()):
                    if dsdir.is_dir() or dsdir.suffix != '.nc':
                        continue
                    if '_mapper' in dsdir.stem:
                        ds = load_dataset(dsdir)
                        mapper = MosaicImageMapper(
                            source_x=ds['source_x'].values,
                            source_y=ds['source_y'].values,
                            target_x=ds['target_x'].values,
                            target_y=ds['target_y'].values,
                            pixel_size=(ds.attrs['pixel_size_x'], ds.attrs['pixel_size_y']),
                            bounds_x=(ds.attrs['bounds_x_0'], ds.attrs['bounds_x_1']),
                            bounds_y=(ds.attrs['bounds_y_0'], ds.attrs['bounds_y_1']),
                            transform=TransformMatrix.from_matrix(ds['transform_matrix'].values),
                        )
                    elif 'window_name' in load_dataset(dsdir).attrs:
                        ds = load_dataset(dsdir)
                        win_name = str(ds.attrs['window_name'])
                        imaps.setdefault(win_name, []).append(ds)
                if mapper is None:
                    raise ValueError('MosaicImageMapper not found in archive')
                if not imaps:
                    raise ValueError('No window datasets found in archive')
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
    ) -> MappedImage:
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
        mapped = MappedImage(
            image=da,
            pixel_size=px
        )
        mapped._imaps = self._imaps
        mapped._windows = self._windows
        return mapped


@dataclass
class MappedImage:
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
                out = DataArray(data*10, dims=('y', 'wavelength'), coords={
                    'y': (
                        ('y',),
                        ds['wly'].data,
                        {
                            'units': 'mm',
                            'description': 'Height in the mosaic coordinate, increasing from the bottom.'
                        }
                    ),
                    'wavelength': (
                        'wavelength',
                        ds['wavelength'].data / 10.0,
                        {
                            'units': 'nm',
                            'description': 'Wavelength in nanometers',
                        }
                    ),
                    'y_slit': (
                        ('y',),
                        ds['y_slit'].data,
                        {
                            'units': 'mm',
                            'description': 'Slit Y coordinate corresponding to mosaic Y',
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
