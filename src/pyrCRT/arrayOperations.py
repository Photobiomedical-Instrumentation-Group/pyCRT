from typing import Any, Callable, List, Tuple, Union

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


def shiftArr(timeArr: np.ndarray, arr: np.ndarray, **kwargs: Any) -> ArrayTuple:
    # {{{
    fromTime = kwargs.get("fromTime", "channel max")
    if fromTime == "channel max":
        fromIndex = np.argmax(arr)
    elif isinstance(fromTime, (int, float)):
        fromIndex = np.where(timeArr >= fromTime)[0][0]

    toTime = kwargs.get("toTime", "end")
    if toTime == "end":
        toIndex = -1
    elif isinstance(toTime, (int, float)):
        toIndex = np.where(timeArr >= toTime)[0][0]

    fromIndex += np.argmax(arr[fromIndex:toIndex])

    return (
        timeArr[fromIndex:toIndex] - np.amin(timeArr[fromIndex:toIndex]),
        arr[fromIndex:toIndex] - np.amin(arr[fromIndex:toIndex]),
    )
    # }}}


def sliceFromMaxToEnd(xArr: Array, yArr: Array) -> ArrayTuple:
    arrSlice = slice(yArr.argmax(), -1)
    return (xArr[arrSlice], yArr[arrSlice])


def sliceByTime(xArr: Array, yArr: Array, fromTime: Real, toTime: Real) -> slice:
    # Kinda silly name for this variable but well, that's what it does. And * is used as
    # a logical "and".
    selectionArray = (yArr >= fromTime) * (yArr <= toTime)
    return (xArr[selectionArray], yArr[selectionArray])


def minMaxNormalize(array: np.ndarray) -> np.ndarray:
    return (array - array.min()) / (array.max() - array.min())


def stripArr(timeArr: np.ndarray, arr: np.ndarray) -> ArrayTuple:
    # {{{
    """Ridiculous workaround for mp4 files"""

    timeArr = np.trim_zeros(timeArr, trim="b")
    arr = arr[: len(timeArr)]
    return timeArr, arr


# }}}
