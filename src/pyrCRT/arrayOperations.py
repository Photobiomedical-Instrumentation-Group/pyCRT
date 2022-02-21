"""
Simple functions for manipulating arrays (pyrCRT.arrayOperations)

This module is mostly functions that receive two arrays as input and returns two arrays,
intended to be used with the timeScds and avgInten arrays.
"""

from typing import Optional, Tuple, Union

import numpy as np

# Type aliases for commonly used types
# {{{
# Used just as a shorthand
Array = np.ndarray

# Tuples of two numpy arrays, typically an array of the timestamp for each frame and an
# array of average intensities within a given ROI
ArrayTuple = Tuple[Array, Array]

Real = Union[float, int, np.float_, np.int_]
# }}}


def sliceFromMaxToEnd(xArr: Array, yArr: Array) -> ArrayTuple:
    # {{{
    # {{{
    """
    Slices the input arrays from the index of the second array's absolute maximum to
    their end.

    Parameters
    ----------
    xArr, yArr : np.ndarray
        The arrays to be sliced. Only the absolute maximum of yArr will be considered.

    Returns
    -------
    xArrSliced, yArrSliced : np.ndarray
        A view to the input arrays, but sliced.
    """
    # }}}
    arrSlice = slice(yArr.argmax(), -1)
    return (xArr[arrSlice], yArr[arrSlice])


# }}}


def sliceByTime(
    xArr: Array,
    yArr: Array,
    fromTime: Optional[Real] = None,
    toTime: Optional[Real] = None,
) -> ArrayTuple:
    # {{{
    # {{{
    """
    Slices the input arrays from the first index whose xArr value is greater than
    fromTime, up to the first index whose xArr value is less than or equal to toTime.

    Parameters
    ----------
    xArr, yArr : np.ndarray
        The arrays to be sliced. Only xArr is actually used for calculating the slice.

    fromTime : real number or None, default=None
        xArr and yArr will be sliced from the first index whose xArr's value is greater
        than this argument. If None, -np.inf will be used.

    fromTime : real number
        xArr and yArr will be sliced up to the first index whose xArr's value is less
        than or equal to this argument. If None, np.inf will be used.

    Returns
    -------
    xArrSliced, yArrSliced : np.ndarray
        A view to the input arrays, but sliced.
    """
    # }}}

    if fromTime is None:
        fromTime = -np.inf
    if toTime is None:
        toTime = np.inf

    # * is used as a logical And here
    selectionArray = (xArr >= fromTime) * (xArr <= toTime)
    return (xArr[selectionArray], yArr[selectionArray])


# }}}


def sliceFromLocalMax(
    xArr: Array,
    yArr: Array,
    fromTime: Optional[Real] = None,
    toTime: Optional[Real] = None,
) -> ArrayTuple:
    # {{{
    """
    Applies sliceByTime and sliceFromMaxToEnd respectively to the input arrays with
    fromTime and toTime as arguments. Refer to sliceByTime's and sliceFromMaxToEnd's
    docstrings for more information.
    """
    # }}}
    return sliceFromMaxToEnd(*sliceByTime(xArr, yArr, fromTime, toTime))


def minMaxNormalize(array: np.ndarray) -> np.ndarray:
    # {{{
    """Performs min-max normalization on array."""
    return (array - array.min()) / (array.max() - array.min())


# }}}


def stripArr(timeArr: np.ndarray, arr: np.ndarray) -> ArrayTuple:
    # {{{
    """Ridiculous workaround for mp4 files. Simply removes the trailing zeros from
    timeArr and the corresponding arr elements."""

    timeArr = np.trim_zeros(timeArr, trim="b")
    arr = arr[: len(timeArr)]
    return timeArr, arr


# }}}
