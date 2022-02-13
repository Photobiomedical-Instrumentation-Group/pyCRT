"""
Video reading functionality (pyrCRT.videoReading)

This module contains everything directly related to extracting lightly processed data
from video files, in particular obtaining the arrays with each frame's time within the
video and the average intensities array calculated within a region of interest of each
frame.

Notes
-----
    This module has only been tested with wmv and mp4 files, with WMV2, and DIVX and
    MP4V codecs respectively.
"""


from contextlib import contextmanager
from os.path import exists
from typing import List, Optional, Tuple, Union

import cv2 as cv  # pylint: disable=import-error
import numpy as np

# Type aliases for commonly used types
# {{{
# Standard ROI tuple used by OpenCV
RoiTuple = Tuple[int, int, int, int]

# Either a RoiTuple, or "all"
RoiType = Union[RoiTuple, str]

# Tuples of two numpy arrays, typically an array of the timestamp for each frame and an
# array of average intensities within a given ROI
ArrayTuple = Tuple[np.ndarray, np.ndarray]

Real = Union[float, int]
# }}}


def readVideo(
    videoPath: str,
    roi: Optional[RoiType] = None,
    rescaleFactor: Real = 1.0,
    displayVideo: bool = True,
) -> ArrayTuple:
    # {{{
    # {{{
    """
    Extracts the time in seconds of each frame and the average pixel intensities of each
    channel within the region of interest (ROI).

    Parameters
    ----------
    videoPath : str
        Path to the video file in the filesystem.
    roi : Tuple[int, int, int, int] or "all"
        The region of interest, inside which the average of each pixel will be computed.
        This tuple must contain 4 integers: (x, y, length_x, and length_y), where x and
        y are the coordinates of the rectangle's center. If roi == "all", then the
        average will be computed on the entire frame. You can also select the ROI at any
        time during the video by pressing the space bar and dragging the square around
        the desired region.
    rescaleFactor : float, optional
        Factor by which each frame will be scaled. This can help reduce the load on the
        hardware and speed up computation. By default the video won't be scaled.
    displayVideo : bool, optional
        Whether or not to display the video while the processing occurs. This has to be
        set to True if roi is set to None

    Returns
    -------
    timeScdsArr : 1D np.ndarray
        Time in seconds of each frame in the video.
    avgIntenArr : 2D np.ndarray
        Average pixel intensity inside the region of interest with respect to time. It
        is an array with shape (len(timeScdsArr), 3), wherein each element is an array
        with the average intensity of the B, G and R (respectively) channels at that
        instant.

    Raises
    ------
    TypeError
        If roi isn't a tuple or a str.
    ValueError
        If roi is a string, but not "all", or isn't a tuple of 4 elements
    RuntimeError
        If this function finished reading the video but no ROI was passed or selected.
    """
    # }}}

    if isinstance(roi, (tuple, str)):
        if not (len(roi) == 4 or roi == "all"):
            raise ValueError(
                "Invalid value for the ROI. The roi parameter should be either a tuple"
                "of 4 ints or 'all' to use the entire frame."
            )
    else:
        raise TypeError(
            "Invalid type for the ROI. The roi parameter should be either a tuple of 4"
            "ints or 'all' to use the entire frame."
        )

    # Yup, I just assume the ROI is valid if it's a tuple of 4 elements. I'll probably
    # have to change this later.

    timeScdsList: List[float] = []
    avgIntenList: List[float] = []

    with videoCapture(videoPath) as cap:
        for frame in frameReader(cap, rescaleFactor):
            if roi is not None:
                channelsAvgInten = calcAvgInten(frame, roi)
                timeScds = cap.get(cv.CAP_PROP_POS_MSEC) / 1000.0
                avgIntenList.append(channelsAvgInten)
                timeScdsList.append(timeScds)

            if displayVideo:
                frame = drawRoi(frame, roi)
                cv.imshow(videoPath, frame)
                key = cv.waitKey(1)

                if key == ord(" "):
                    roi = cv.selectROI(videoPath, frame)
                    print(f"Selected ROI: {roi}")
                    avgIntenList, timeScdsList = [], []
                elif key == ord("q"):
                    break

    if not avgIntenList:
        raise RuntimeError(
            "Array of average intensities is empty! Did you select or pass"
            "as an argument a region of interest (roi)?"
        )

    avgIntenArr = np.array(avgIntenList)
    timeScdsArr = np.array(timeScdsList)

    timeScdsArr, avgIntenArr = stripArr(timeScdsArr, avgIntenArr)

    return timeScdsArr, avgIntenArr

# }}}


@contextmanager
def videoCapture(*args, **kwargs):
    # {{{
    """Trivial context manager for safely releasing the OpenCV capture after finishing
    to work with it. Performs a sanity check to make sure the make sure the first arg
    has a valid type (int for cameras and str for video paths in the filesystem), and
    that the video path at least exists (when applicable, of course)."""

    if isinstance(args[0], int):
        # Assumes it is a valid camera. There isn't really a way to check this, as far
        # as I know.
        pass
    elif isinstance(args[0], str):
        if not exists(args[0]):
            raise ValueError(
                f"The path ({args[0]}) passed to the videoCapture context manager"
                "does not exist."
            )
    else:
        raise TypeError(
            f"Invalid type of {args[0]} passed as the first argument"
            "to videoCapture. Valid types: int (for a video capture device) and"
            "str (for a path to a video in the filesystem."
        )

    cap = cv.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()
        cv.destroyAllWindows()


# }}}


def frameReader(capture: cv.VideoCapture, rescaleFactor: Real = 1.0) -> np.ndarray:
    # {{{
    # {{{
    """
    A generator for reading each frame from the VideoCapture instance, and optionally
    rescaling it.

    Parameters
    ----------
    capture : cv.VideoCapture
    rescaleFactor : int or float, optional

    Yields
    ------
    frame : np.ndarray
        Width by Length by 3 matrix of BGR pixel intensities
    """
    # }}}

    while True:
        status, frame = capture.read()

        if status:
            frame = rescaleFrame(frame, rescaleFactor)
            yield frame
        else:
            break


# }}}


def rescaleFrame(frame: np.ndarray, rescaleFactor: Real) -> np.ndarray:
    # {{{
    """Simply rescales the frame, uses bilinear interpolation."""

    return cv.resize(frame, (0, 0), fx=rescaleFactor, fy=rescaleFactor)


# }}}


def drawRoi(frame: np.ndarray, roi: RoiType) -> np.ndarray:
    # {{{
    """Simply draws a red rectangle on the frame to highlight the ROI"""

    if isinstance(roi, tuple):
        x1, y1, sideX, sideY = roi
        x2, y2 = x1 + sideX, y1 + sideY
        return cv.rectangle(frame, (x1, y1), (x2, y2), [0, 0, 255], 2)
    return frame # in this case, roi will *prolly* be == "all"


# }}}


def cropFrame(frame: np.ndarray, roi: RoiType) -> np.ndarray:
    # {{{
    """Slices the frame matrix and returns only the portion inside the ROI"""

    if isinstance(roi, tuple):
        x1, y1, sideX, sideY = roi
        return frame[y1 : y1 + sideY, x1 : x1 + sideX]
    return frame # in this case, roi will *prolly* be == "all"


# }}}


def calcAvgInten(frame: np.ndarray, roi: RoiType) -> np.ndarray:
    # {{{
    """Calculates the average pixel intensity for all pixels inside the ROI and returns
    an array with the average for each channel."""

    croppedFrame = cropFrame(frame, roi)
    channelsAvgInten = cv.mean(croppedFrame)[:3]
    return channelsAvgInten


# }}}

# TODO: Move these functions to .arrayOperations as soon as you're finished testing this
# module.

def stripArr(timeArr: np.ndarray, arr: np.ndarray) -> ArrayTuple:
    # {{{
    """Ridiculous workaround for mp4 files"""

    timeArr = np.trim_zeros(timeArr, trim="b")
    arr = arr[: len(timeArr)]
    return timeArr, arr


# }}}
