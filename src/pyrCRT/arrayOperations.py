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


def sliceFromMaxToEnd(intenArr: Array) -> slice:
    # {{{
    # {{{
    """
    Returns a slice object that slices the input array from the index of its absolute
    maximum to its end. It's literally just slice(intenArr.argmax(), -1)
    """
    # }}}
    return slice(intenArr.argmax(), -1)


# }}}


def sliceByTime(
    timeArr: Array,
    fromTime: Optional[Real] = None,
    toTime: Optional[Real] = None,
) -> slice:
    # {{{
    # {{{
    """
    Creates a slice object specifying timeArr's section which is between fromTime and
    toTime.

    Parameters
    ----------
    timeArr : np.ndarray
        The array whose slice is to be calculated

    fromTime : real number or None, default=None
        The slice's "start" argument will be the index of the first element of timeArr
        whose value is greater or equal to fromTime. If None, -np.inf will be used.

    toTime : real number or None, default=None
        The slice's "stop" argument will be the index of the first element of timeArr
        whose value is less or equal to fromTime. If None, np.inf will be used.

    Returns
    -------
    A slice object specifying the slice between fromTime to toTime in arr.
    """
    # }}}

    if fromTime is None:
        fromTime = -np.inf
    if toTime is None:
        toTime = np.inf

    # * is used as a logical And here
    selectionArray = ((timeArr >= fromTime) * (timeArr <= toTime)).nonzero()[0]
    fromIndex, toIndex = selectionArray[0], selectionArray[-1]
    return slice(fromIndex, toIndex)


# }}}


def sliceFromLocalMax(
    timeArr: Array,
    intenArr: Array,
    fromTime: Optional[Real] = None,
    toTime: Optional[Real] = None,
) -> slice:
    # {{{
    # {{{
    """
    Applies sliceByTime and sliceFromMaxToEnd in this order to the input arrays with
    fromTime and toTime as arguments. Refer to sliceByTime's and sliceFromMaxToEnd's
    docstrings for more information.
    """
    # }}}
    timeSlice = sliceByTime(timeArr, fromTime, toTime)
    timeFrom, timeTo, _ = timeSlice.indices(len(timeArr))

    fromMaxSlice = sliceFromMaxToEnd(intenArr[timeSlice])
    maxIndex, _, _ = fromMaxSlice.indices(len(timeArr))
    return slice(timeFrom + maxIndex, timeTo)


# }}}


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
