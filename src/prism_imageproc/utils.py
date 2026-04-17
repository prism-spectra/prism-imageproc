# %% Imports
from __future__ import annotations
from typing import Any, List, Optional, Union
from dacite import Config
from tomlkit import register_encoder
from tomlkit.items import Item as TomlItem, item as tomlitem
from astropy.units import Quantity
import astropy.units as u
from numpy import ndarray, asarray, fromstring
from numpy.typing import NDArray
from json import JSONEncoder
import sys
# %% Type Aliases
if sys.version_info >= (3, 11):
    type MaybeQuantity = str | Quantity
else:
    from typing import TypeAlias, Union
    MaybeQuantity: TypeAlias = Union[str, Quantity]


def to_quantity(value: MaybeQuantity) -> Quantity:
    if isinstance(value, str):
        return Quantity(value)
    elif isinstance(value, Quantity):
        return value
    else:
        raise ValueError('Invalid quantity specification.')


def optional_quantity(value: Optional[MaybeQuantity]) -> Optional[Quantity]:
    if value is None:
        return None
    return to_quantity(value)

# %% Serde Helpers


@register_encoder
def qty_ndarray_encoder(obj: Any, /, _parent=None, _sort_keys=False) -> TomlItem:
    if isinstance(obj, Quantity):
        return tomlitem(f'{obj}')
    elif isinstance(obj, ndarray):
        return tomlitem(obj.tolist())
    else:
        raise TypeError(
            f'Object of type {type(obj)} is not JSON serializable.')


class QuantityEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Quantity):
            return f'{o}'
        elif isinstance(o, ndarray):
            return o.tolist()
        else:
            return super().default(o)


class QuantityDecoder:
    @staticmethod
    def decode_qty(value: str) -> Quantity:
        value = value.strip()
        if value.startswith('['):
            arr_str, unit = value.rsplit(']', 1)
            arr_str = arr_str.strip('[]')
            arr = fromstring(arr_str, sep=' ', dtype=float)
            return Quantity(arr, unit.strip())
        return Quantity(value)

    @staticmethod
    def decode_ndarray(value: List[Any]) -> NDArray:
        return asarray(value, dtype=float)

    @property
    def config(self) -> Config:
        return Config(
            type_hooks={
                Quantity: self.decode_qty,
                ndarray: self.decode_ndarray
            },
        )


QUANTITY_DECODER = QuantityDecoder().config