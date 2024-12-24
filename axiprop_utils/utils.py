import typing
import numpy as np

if typing.TYPE_CHECKING:
    import pint
    from typing import Tuple


def __is_sequence_size_2(arr):
    return isinstance(arr, typing.Sequence) and len(arr) == 2


def ensure_tuple(arr):
    """Ensure that the input is a tuple of size 2.

    Parameters
    ----------
    arr : float or tuple
        The input to be converted to a tuple

    Returns
    -------
    tuple
        The input as a tuple of size 2: (arr, arr) if arr is a float, or arr if it is already a tuple of size 2
    """
    if __is_sequence_size_2(arr):
        return arr
    return arr, arr


def generate_r_axis(
    x_axis: 'pint.Quantity | np.ndarray', y_axis: 'pint.Quantity | np.ndarray'
) -> 'Tuple[np.ndarray, np.ndarray, np.ndarray]':
    """Give the radial axis from the x and y axes as required by ScalarFieldEnvelope import.

    Parameters
    ----------
    x_axis : pint.Quantity | np.ndarray
        x-axis array
    y_axis : pint.Quantity | np.ndarray
        y-axis array

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        a tuple of (r_axis, x_axis, y_axis) where r_axis is the radial axis
    """
    from pic_utils.units import strip_units

    x_axis = strip_units(x_axis, 'm')
    y_axis = strip_units(y_axis, 'm')

    r_axis = np.sqrt(x_axis[:, None] ** 2 + y_axis[None, :] ** 2)

    return r_axis, x_axis, y_axis
