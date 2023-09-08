"""
Simple functions for manipulating arrays (pyCRT.arrayOperations)

This module is mostly functions that receive two arrays as input and returns
two arrays, intended to be used with the timeScds and avgInten arrays.
"""

from typing import Any, Optional, Union
from warnings import warn

import numpy as np

# pylint: disable=no-name-in-module,import-error
from numpy.typing import NDArray

# Type aliases for commonly used types
# {{{
# Array of arbitraty size with float elements.
Array = NDArray[np.float_]

# Tuples of two numpy arrays, typically an array of the timestamp for each
# frame and an array of average intensities within a given ROI
ArrayTuple = tuple[Array, Array]

Real = Union[float, int, np.float_, np.int_]
# }}}


def sliceFromMaxToEnd(intenArr: Array) -> slice:
    # {{{
    # {{{
    """
    Returns a slice object that slices the input array from the index of its
    absolute maximum to its end. It's literally just slice(intenArr.argmax(),
    -1)
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

    Creates a slice object specifying timeArr's section which is between
    fromTime and toTime.

    Parameters
    ----------
    timeArr : np.ndarray
        The array whose slice is to be calculated

    fromTime : real number or None, default=None
        The slice's "start" argument will be the index of the first element of
        timeArr whose value is greater or equal to fromTime. If None, -np.inf
        will be used.

    toTime : real number or None, default=None
        The slice's "stop" argument will be the index of the first element of
        timeArr whose value is less or equal to fromTime. If None, np.inf will
        be used.

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
    Applies sliceByTime and sliceFromMaxToEnd (in this order) to the input
    arrays with fromTime and toTime as arguments, such that the resulting slice
    starts at intenArr's maximum value and ends at toTime.
    """

    # }}}
    timeSlice = sliceByTime(timeArr, fromTime, toTime)
    timeFrom, timeTo, _ = timeSlice.indices(len(timeArr))

    fromMaxSlice = sliceFromMaxToEnd(intenArr[timeSlice])
    maxIndex, _, _ = fromMaxSlice.indices(len(timeArr))
    return slice(timeFrom + maxIndex, timeTo)


# }}}


def minMaxNormalize(array: Array) -> np.ndarray:
    # {{{
    """Performs min-max normalization on array."""
    return (array - array.min()) / (array.max() - array.min())


# }}}


def stripArr(timeArr: Array, arr: Array) -> ArrayTuple:
    # {{{
    """Ridiculous workaround for mp4 files. Simply removes the trailing zeros
    from timeArr and the corresponding arr elements."""

    timeArr = np.trim_zeros(timeArr, trim="b")
    arr = arr[: len(timeArr)]
    return timeArr, arr


# }}}


def subtractMinimum(arr: Array) -> Array:
    # {{{
    """
    Subtracts the array's elements by the array's minimum value. What else did
    you expect?
    """

    return arr - arr.min()


# }}}


# You should look into mypy generics to avoid the value: Any here
def findValueIndex(arr: Array, value: Any) -> int:
    # {{{
    """Returns the index of the first element in arr which is greater than
    value."""
    try:
        index = int(np.where(arr >= float(value))[0][0])
        valueRatio = abs(arr[index] / value)
        if valueRatio > 1.5:
            warn(
                f"The array's closest value greater than {value} is "
                f"{arr[index]},  which is {100*valueRatio:.0f}% the "
                "specified value. This may not be what you want."
            )
        return int(np.where(arr >= float(value))[0][0])
    except IndexError as err:
        raise IndexError(
            f"No value in arr is greater or equal than {value}"
        ) from err


# }}}
