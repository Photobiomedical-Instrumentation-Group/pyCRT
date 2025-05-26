"""
Frame manipulation functions (pyCRT.frameOperations)

A collection of small functions that that do something to frames (which are 3D
numpy arrays that represent each channel of each pixel of an image), typically
recieving a frame and returning another. Functionality includes cropping,
calculating average intensity, drawing a ROI, etc,
"""

from typing import Optional, Union

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

# Type aliases for commonly used types
# {{{
# Array of arbitraty size with float elements.
Array = NDArray[np.float64]

# Standard ROI tuple used by OpenCV
RoiTuple = tuple[int, int, int, int]

# Either a RoiTuple, or "all"
RoiType = Union[RoiTuple, str]

Real = Union[float, int, np.float64, int]
# }}}


def drawRoi(frame: Array, roi: Optional[RoiType]) -> Array:
    # {{{
    """
    Draws a red rectangle around the ROI.

    Parameters
    ----------
    frame : 3D np.ndarray
        A matrix with each channel's intensity for each pixel. The output of
        cv2.imread.

    roi : tuple of 4 ints, str or None
        The ROI. It can be a tuple of 4 ints, in which case the first two are
        the x and y coordinates of the rectangle's top-left corner, and the
        other two are the lengths of its sides. If not a tuple, it'll just
        assume the ROI hasn't been specified yet or is supposed to be the
        entire frame, in which case the original frame will be returned
        unmodified.

    Returns
    -------
    frameWithRoi : 3D np.ndarray
        The original frame passed to this function, either unmodified or with
        the sides of a red rectangle drawn on the coordinates specified by the
        roi tuple.
    """

    if isinstance(roi, tuple):
        x1, y1, sideX, sideY = roi
        x2, y2 = x1 + sideX, y1 + sideY
        return cv.rectangle(frame, (x1, y1), (x2, y2), [0, 0, 255], 2)
    return frame  # in this case, roi will *prolly* be == "all", or None


# }}}


def cropFrame(frame: Array, roi: Optional[RoiType]) -> Array:
    # {{{
    """
    Draws a red rectangle around the ROI.

    Parameters
    ----------
    frame : 3D np.ndarray
        A matrix with each channel's intensity for each pixel. The output of
        cv2.imread.

    roi : tuple of 4 ints, str or None
        The ROI. It can be a tuple of 4 ints, in which case the first two are
        the x and y coordinates of the rectangle's top-left corner, and the
        other two are the lengths of its sides. If not a tuple, it'll just
        assume the ROI hasn't been specified yet or is supposed to be the
        entire frame, in which case the original frame will be returned
        unmodified.

    Returns
    -------
    croppedFrame : 3D np.ndarray
        The portion of the frame inside the rectangle specified by the roi
        parameter. This array's dimensions will be the last two elements of the
        roi tuple. If roi is not a tuple, it'll just return the entire frame.
    """

    if isinstance(roi, tuple):
        x1, y1, sideX, sideY = roi
        return frame[y1 : y1 + sideY, x1 : x1 + sideX]
    return frame  # in this case, roi will *prolly* be == "all", or None


# }}}


def calcAvgInten(frame: Array, roi: Optional[RoiType], gamma: float) -> Array:
    # {{{
    """
    Calculates the average channel intensity on the pixels inside the ROI.

    Parameters
    ----------
    frame : 3D np.ndarray
        A matrix with each channel's intensity for each pixel. The output of
        cv2.imread.

    roi : tuple of 4 ints, str or None
        The ROI. It can be a tuple of 4 ints, in which case the first two are
        the x and y coordinates of the rectangle's top-left corner, and the
        other two are the lengths of its sides. If not a tuple, the average
        will be computed over the entire frame.

    gamma : float
        A parameter represents the gamma correction value, which is used to
        adjust the brightness of an image. This value is obtained by comparing
        the color intensity values from a standard color chart (a reference
        using Colorimeter) with the actual RGB values in the image. The goal is
        to correct for non-linearities in how colors are captured or displayed,
        ensuring that the output image accurately represents the original
        scene's colors.

      def gamma_fit(x, a, b, c):
        result = (a * x) ** b + c
             return result

    Here, x represents the input pixel values, and a, b, and c are parameters
    derived from the fitting process. The key component is b, where 1/b gives
    us the gamma correction value. This value is used to correct the image's
    brightness and contrast.

    Returns
    -------
    channelsAvgInten : 3x1 np.ndarray
        The average intensity for each channel (in the order of BGR),
        calcualted with the pixels inside the ROI.
    """

    croppedFrame = cropFrame(frame, roi)
    channelsAvgInten = cv.mean(croppedFrame)[:3]

    # sem correção Gamma:
    channelsAvgIntenArray = np.array(channelsAvgInten)

    # Correção gamma
    channelsAvgIntenCorrigido = (channelsAvgIntenArray / 255.0) ** gamma * 255

    return channelsAvgIntenCorrigido


# }}}


# Frame funcs for readVideo


def doNothing(frame):
    return frame


def rescaleFrame(frame: Array, rescaleFactor: Real) -> Array:
    # {{{
    """
    Changes the frame's dimensions, using bilinear interpolation.

    Parameters
    ----------
    frame : 3D np.ndarray
        A matrix with each channel's intensity for each pixel. The output of
        cv2.imread.

    rescaleFactor : real number
        The factor by which the frame's dimensions will be multiplied.

    Returns
    -------
    rescaledFrame : 3D np.ndarray
        The rescaled frame, using bilinear interpolation.
    """
    rescaleFactor = float(rescaleFactor)
    return cv.resize(frame, (0, 0), fx=rescaleFactor, fy=rescaleFactor)


# }}}


def colorConvertFactory(to_space, from_space="BGR", to_f32bits=True):
    # {{{
    cvAttrName = (
        f"COLOR_{from_space.strip().upper()}2{to_space.strip()}"
    )
    cvAttr = getattr(cv, cvAttrName)

    if to_f32bits:

        def convertFunc(frame):
            frame = (frame / 255).astype(np.float32)
            return cv.cvtColor(frame, cvAttr)

    else:

        def convertFunc(frame):
            return cv.cvtColor(frame, cvAttr)

    return convertFunc


# }}}


bgrToHls = colorConvertFactory("HLS")
bgrToHsv = colorConvertFactory("HSV")
bgrToLab = colorConvertFactory("LAB")
bgrToYuv = colorConvertFactory("YUV")
bgrToLuv = colorConvertFactory("LUV")
bgrToYCrCb = colorConvertFactory("YCrCb")
bgrToGray = colorConvertFactory("GRAY")
