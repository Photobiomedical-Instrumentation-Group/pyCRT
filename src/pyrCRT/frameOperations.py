"""
Frame manipulation functions (pyrCRT.frameOperations)

A collection of small functions that that do something to frames (which are 3D numpy
arrays that represent each channel of each pixel of an image), typically recieving a
frame and returning another. Functionality includes cropping, calculating average
intensity, drawing a ROI, etc,
"""

from typing import Optional, Union

# pylint: disable=import-error
import cv2 as cv  # type: ignore
import numpy as np

# pylint: disable=no-name-in-module,import-error
from numpy.typing import NDArray

# Type aliases for commonly used types
# {{{
# Array of arbitraty size with float elements.
Array = NDArray[np.float_]

# Standard ROI tuple used by OpenCV
RoiTuple = tuple[int, int, int, int]

# Either a RoiTuple, or "all"
RoiType = Union[RoiTuple, str]

Real = Union[float, int, np.float_, np.int_]
# }}}


def rescaleFrame(frame: Array, rescaleFactor: Real) -> Array:
    # {{{
    """Simply rescales the frame, uses bilinear interpolation."""

    return cv.resize(frame, (0, 0), fx=rescaleFactor, fy=rescaleFactor)


# }}}


def drawRoi(frame: Array, roi: Optional[RoiType]) -> Array:
    # {{{
    """Simply draws a red rectangle on the frame to highlight the ROI"""

    if isinstance(roi, tuple):
        x1, y1, sideX, sideY = roi
        x2, y2 = x1 + sideX, y1 + sideY
        return cv.rectangle(frame, (x1, y1), (x2, y2), [0, 0, 255], 2)
    return frame  # in this case, roi will *prolly* be == "all", or None


# }}}


def cropFrame(frame: Array, roi: Optional[RoiType]) -> Array:
    # {{{
    """Slices the frame matrix and returns only the portion inside the ROI"""

    if isinstance(roi, tuple):
        x1, y1, sideX, sideY = roi
        return frame[y1 : y1 + sideY, x1 : x1 + sideX]
    return frame  # in this case, roi will *prolly* be == "all", or None


# }}}


def calcAvgInten(frame: Array, roi: Optional[RoiType]) -> Array:
    # {{{
    """Calculates the average pixel intensity for all pixels inside the ROI and returns
    an array with the average for each channel."""

    croppedFrame = cropFrame(frame, roi)
    channelsAvgInten = cv.mean(croppedFrame)[:3]
    return channelsAvgInten


# }}}
