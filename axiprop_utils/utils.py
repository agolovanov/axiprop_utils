import typing


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
