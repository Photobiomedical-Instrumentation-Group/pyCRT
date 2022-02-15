"""
Video reading functionality (pyrCRT.videoReading)

This module contains everything directly related to extracting lightly processed data
from video files or cameras, in particular obtaining the arrays with each frame's time
within the video and the average intensities array calculated within a region of
interest of each frame.

Notes
-----
    This module has only been tested with wmv and mp4 files, with WMV2, and DIVX and
    MP4V codecs respectively.
"""


from contextlib import contextmanager
from os.path import exists
from typing import Any, Generator, List, Optional, Tuple, Union

# pylint: disable=import-error
import cv2 as cv  # type: ignore
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

Real = Union[float, int, np.float_, np.int_]
# }}}


def readVideo(
    videoSource: Union[str, int],
    roi: Optional[RoiType] = None,
    displayVideo: bool = True,
    recordingPath: Optional[str] = None,
    rescaleFactor: Real = 1.0,
    cameraResolution: Optional[Tuple[int, int]] = None,
    codecFourcc: str = "mp4v",
    recordingFps: float = 30.0,
    **kwargs: Any,
) -> ArrayTuple:
    # {{{
    # {{{
    """
    Extracts the time in seconds of each frame and the average pixel intensities of each
    channel within the region of interest (ROI).

    Parameters
    ----------
    videoSource : int or str
        The first argument to cv2.VideoCapture. If an int, it will take the
        corresponding detected camera as the video source. If a str, It'll take it as a
        path in the filesystem for a video file.
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

    cameraResolution : tuple of 2 ints, default=None
        Used to optionally change the camera resolution before handing over the
        VideoCapture instance. If reading from a video file, it does nothing.

    recordingPath : str, default=None
        The path (with the extension!) in the filesystem wherein to save the recording.
        If falsy, it won't record the video.
    codecFourcc : str, default='mp4v'
        The fourcc identifier for the video codec to be used for the recording. Refer to
        www.fourcc.org/codecs.php for a list of possible codes.
    recordingFps : int, default=None
        The FPS (frames per second) for the recording, which doesn't need to correspond
        to the FPS of the camera or the source video.

    kwargs : Any
        Additional arguments to be passed to cv2.VideoCapture

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

    if recordingPath:
        writer = frameWriter(recordingPath, codecFourcc, recordingFps)
        writer.send(None)  # Initialize generator

    # Yup, I just assume the ROI is valid if it's a tuple of 4 elements. I'll probably
    # have to change this later.

    timeScdsList: List[float] = []
    avgIntenList: List[np.ndarray] = []

    with videoCapture(videoSource, cameraResolution, **kwargs) as cap:
        for frame in frameReader(cap, rescaleFactor):
            if roi is not None:
                timeScds = cap.get(cv.CAP_PROP_POS_MSEC) / 1000.0
                channelsAvgInten = calcAvgInten(frame, roi)
                timeScdsList.append(timeScds)
                avgIntenList.append(channelsAvgInten)

            if recordingPath:
                writer.send(frame)

            if displayVideo:
                frame = drawRoi(frame, roi)
                cv.imshow("Video stream", frame)
                key = cv.waitKey(1)

                if key == ord(" "):
                    roi = cv.selectROI("Video stream", frame)
                    print(f"Selected ROI: {roi}")
                    timeScdsList, avgIntenList = [], []
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
def videoCapture(
    videoSource: Union[int, str],
    cameraResolution: Optional[Tuple[int, int]] = None,
    **kwargs: Any,
) -> cv.VideoCapture:
    # {{{
    # {{{
    """
    Context manager for working with a cv2.VideoCapture instance.

    Parameters
    ----------
    videoSource : int or str
        The first argument to cv2.VideoCapture. If an int, it will take the
        corresponding detected camera as the video source. If a str, It'll take it as a
        path in the filesystem for a video file.
    cameraResolution : tuple of 2 ints, default=None
        Used to optionally change the camera resolution before handing over the
        VideoCapture instance. If reading from a video file, it does nothing.
    **kwargs : any type
        Additional arguments to pass to cv2.VideoCapture

    Yields
    ------
    cv2.VideoCapture

    Raises
    ------
    ValueError
        If videoSource is a str but there isn't any file in the specified path.
    TypeError
        If videoSource isn't a str or an int

    Notes
    -----
        This context manager performs no other checks other than those indicated on the
        Raises section. This warrants mentioning because OpenCV gives the most cryptic,
        confusing and even misleading error messages, so care must be taken.
    """
    # }}}

    if isinstance(videoSource, int):
        # Assumes it is a valid camera. There isn't really a way to check this, as far
        # as I know.
        pass
    elif isinstance(videoSource, str):
        if not exists(videoSource):
            raise ValueError(
                f"The path ({videoSource}) passed to the videoCapture context manager"
                "does not exist."
            )
    else:
        raise TypeError(
            f"Invalid type of {type(videoSource)} passed as the first argument"
            "to videoCapture. Valid types: int (for a video capture device) and"
            "str (for a path to a video in the filesystem."
        )

    cap = cv.VideoCapture(videoSource, **kwargs)

    # Set camera resolution
    if cameraResolution:
        resX, resY = cameraResolution
        cap.set(3, resX)
        cap.set(4, resY)

    try:
        yield cap
    finally:
        cap.release()
        cv.destroyAllWindows()


# }}}


def frameReader(
    capture: cv.VideoCapture,
    rescaleFactor: Real = 1.0,
) -> Generator:
    # {{{
    # {{{
    """
    A generator for reading each frame from the VideoCapture instance, and optionally
    rescaling it.

    Parameters
    ----------
    capture : cv.VideoCapture
        The OpenCV VideoCapture instance from which to extract the frames. See
        pyrCRT.videoReading.videoCapture
    rescaleFactor : int or float, default=1.0
        Factor by which each frame will be scaled. This can help reduce the load on the
        hardware and speed up computation. By default the video won't be scaled.

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


def frameWriter(
    recordingPath: str,
    codecFourcc: str = "mp4v",
    recordingFps: float = 30.0,
) -> Generator:
    # {{{
    # {{{
    """
    A generator that acts as a coroutine for recording the capture's frames. The idea is
    that you initialize this generator with the arguments described below and send each
    frame to be recorded with the send method.

    Parameters
    ----------
    recordingPath : str, default=None
        The path (with the extension!) in the filesystem wherein to save the recording.
    codecFourcc : str, default='mp4v'
        The fourcc identifier for the video codec to be used for the recording. Refer to
        www.fourcc.org/codecs.php for a list of possible codes.
    recordingFps : int, default=None
        The FPS (frames per second) for the recording, which doesn't need to correspond
        to the FPS of the camera or the source video.
    """
    # }}}

    # This first recieved frame is used to pass the correct frame dimentions to
    # cv.VideoWriter
    frame = yield
    writer = cv.VideoWriter(
        recordingPath,
        cv.VideoWriter_fourcc(*codecFourcc),
        recordingFps,
        frame.shape[1::-1],
    )
    writer.write(frame)

    while True:
        frame = yield
        writer.write(frame)


# }}}


# TODO: Move these functions to .frameManipulation as soon as you're finished testing
# this module.


def rescaleFrame(frame: np.ndarray, rescaleFactor: Real) -> np.ndarray:
    # {{{
    """Simply rescales the frame, uses bilinear interpolation."""

    return cv.resize(frame, (0, 0), fx=rescaleFactor, fy=rescaleFactor)


# }}}


def drawRoi(frame: np.ndarray, roi: Optional[RoiType]) -> np.ndarray:
    # {{{
    """Simply draws a red rectangle on the frame to highlight the ROI"""

    if isinstance(roi, tuple):
        x1, y1, sideX, sideY = roi
        x2, y2 = x1 + sideX, y1 + sideY
        return cv.rectangle(frame, (x1, y1), (x2, y2), [0, 0, 255], 2)
    return frame  # in this case, roi will *prolly* be == "all", or None


# }}}


def cropFrame(frame: np.ndarray, roi: Optional[RoiType]) -> np.ndarray:
    # {{{
    """Slices the frame matrix and returns only the portion inside the ROI"""

    if isinstance(roi, tuple):
        x1, y1, sideX, sideY = roi
        return frame[y1 : y1 + sideY, x1 : x1 + sideX]
    return frame  # in this case, roi will *prolly* be == "all", or None


# }}}


def calcAvgInten(frame: np.ndarray, roi: Optional[RoiType]) -> np.ndarray:
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
