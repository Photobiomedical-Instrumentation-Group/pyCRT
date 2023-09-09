"""
Video reading functionality (pyCRT.videoReading)

This module contains everything directly related to extracting lightly
processed data from video files or cameras, in particular obtaining the arrays
with each frame's time within the video and the average intensities array
calculated within a region of interest of each frame.

Notes
-----
    This module has only been tested with wmv and mp4 files, with WMV2, and
    DIVX and MP4V codecs respectively.
"""


from contextlib import contextmanager
from os.path import isfile
from time import sleep
from typing import Any, Generator, Iterator, Optional, Sequence, Union
from warnings import warn

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from .arrayOperations import stripArr
from .configFiles import loadTOML
from .frameOperations import calcAvgInten, drawRoi, rescaleFrame

# Type aliases for commonly used types
# {{{
# Array of arbitraty size with float elements.
Array = NDArray[np.float_]

# Standard ROI tuple used by OpenCV
RoiTuple = tuple[int, int, int, int]

# Either a RoiTuple, or "all"
RoiType = Union[RoiTuple, str]

# Tuples of two numpy arrays, typically an array of the timestamp for each
# frame and an array of average intensities within a given ROI
ArrayTuple = tuple[Array, Array]

Real = Union[float, int, np.float_, np.int_]
Integer = Union[int, np.int_]

# Dictionary created from a TOML configuration file
TOMLDict = dict[str, Union[list, int, dict[str, Any]]]
# }}}

# Dictionaries containing openCV's capture device properties for ease of access
NUMERIC_CODES = {  # {{{
    "width": cv.CAP_PROP_FRAME_WIDTH,
    "height": cv.CAP_PROP_FRAME_HEIGHT,
    "brightness": cv.CAP_PROP_BRIGHTNESS,
    "contrast": cv.CAP_PROP_CONTRAST,
    "saturation": cv.CAP_PROP_SATURATION,
    "gain": cv.CAP_PROP_GAIN,
    "WBTemp": cv.CAP_PROP_WB_TEMPERATURE,
    "sharpness": cv.CAP_PROP_SHARPNESS,
    "exposure": cv.CAP_PROP_EXPOSURE,
    "pan": cv.CAP_PROP_PAN,
    "tilt": cv.CAP_PROP_TILT,
    "focus": cv.CAP_PROP_FOCUS,
    "zoom": cv.CAP_PROP_ZOOM,
    "FPS": cv.CAP_PROP_FPS,
    "hue": cv.CAP_PROP_HUE,
}

BOOLEAN_CODES = {
    "autoWBTemp": cv.CAP_PROP_AUTO_WB,
    "autoExposure": cv.CAP_PROP_AUTO_EXPOSURE,
    "autoFocus": cv.CAP_PROP_AUTOFOCUS,
}

ALL_CODES = {**NUMERIC_CODES, **BOOLEAN_CODES}  # }}}

# Dictionary for the property name that corresponds to each openCV prop code
ALL_PROPS = {v: k for k, v in ALL_CODES.items()}


def readVideo(
    videoSource: Union[str, int],
    roi: Optional[RoiType] = None,
    displayVideo: bool = True,
    recordingPath: Optional[str] = None,
    rescaleFactor: Real = 1.0,
    waitKeyTime: int = 1,
    cameraResolution: Optional[tuple[int, int]] = None,
    codecFourcc: str = "mp4v",
    recordingFps: float = 30.0,
) -> ArrayTuple:
    # {{{
    # {{{
    """
    Extracts the time in seconds of each frame and the average pixel
    intensities of each channel within the region of interest (ROI).

    Parameters
    ----------
    videoSource : int or str
        The first argument to cv2.VideoCapture. If an int, it will take the
        corresponding detected camera as the video source. If a str, It'll take
        it as a path in the filesystem for a video file.

    roi : Tuple[int, int, int, int] or "all"
        The region of interest, inside which the average of each pixel will be
        computed. This tuple must contain 4 integers: (x, y, length_x, and
        length_y), where x and y are the coordinates of the rectangle's center.
        If roi == "all", then the average will be computed on the entire frame.
        You can also select the ROI at any time during the video by pressing
        the space bar and dragging the square around the desired region.

    displayVideo : bool, default=True
        Whether or not to display the video while it is being read. This must
        be set to True if no ROI is specified, so the ROI can be manually
        selected by pressing the spacebar during the video exhibition.

    recordingPath : str, default=None
        The path (with the extension!) in the filesystem wherein to save the
        recording. If false, it won't record the video.

    rescaleFactor : real number, optional
        Factor by which each frame will be scaled. This can help reduce the
        load on the hardware and speed up computation. By default the video
        won't be scaled.

    waitKeyTime : int, optional
        How many milliseconds to wait for user input between each frame. The
        default value is 1, so on most machines the video will appear "sped up"
        relative to it being played on a regular video player. See cv2.waitKey
        for more information.

    cameraResolution : tuple of 2 ints, default=None
        Used to optionally change the camera resolution before handing over the
        VideoCapture instance. If reading from a video file, it does nothing.

    codecFourcc : str, default='mp4v'
        The fourcc identifier for the video codec to be used for the recording.
        Refer to www.fourcc.org/codecs.php for a list of possible codes.

    recordingFps : float, default=None
        The FPS (frames per second) for the recording, which doesn't need to
        correspond to the FPS of the camera or the source video.

    Returns
    -------
    fullTimeScdsArr : 1D np.ndarray
        Time in seconds of each frame in the video.
    channelsAvgIntensArr : 2D np.ndarray
        Average pixel intensity inside the region of interest with respect to
        time. It is an array with shape (len(fullTimeScdsArr), 3), wherein each
        element is an array with the average intensity of the B, G and R
        (respectively) channels at that instant.

    Raises
    ------
    TypeError
        If roi isn't a tuple or a str.
    ValueError
        If roi is a string, but not "all", or isn't a tuple of 4 elements
    RuntimeError
        If this function finished reading the video but no ROI was passed or
        selected.
    """
    # }}}

    if isinstance(roi, (tuple, str)):
        if not (len(roi) == 4 or roi == "all"):
            raise ValueError(
                "Invalid value for the ROI. The roi parameter should be "
                "either a tuple of 4 ints or 'all' to use the entire frame."
            )
    elif roi is not None:
        raise TypeError(
            "Invalid type for the ROI. The roi parameter should be either a "
            "tuple of 4 ints or 'all' to use the entire frame."
        )

    if recordingPath:
        writer = frameWriter(recordingPath, codecFourcc, recordingFps)
        # initialize generator
        writer.send(None)  # type: ignore

    # Yup, I just assume the ROI is valid if it's a tuple of 4 elements. I'll
    # probably have to change this later.

    timeScdsList: list[float] = []
    avgIntenList: list[Array] = []

    with videoCapture(videoSource, cameraResolution) as cap:
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
                key = cv.waitKey(waitKeyTime)

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

    channelsAvgIntensArr = np.array(avgIntenList)
    fullTimeScdsArr = np.array(timeScdsList)

    fullTimeScdsArr, channelsAvgIntensArr = stripArr(
        fullTimeScdsArr, channelsAvgIntensArr
    )

    return fullTimeScdsArr - fullTimeScdsArr.min(), channelsAvgIntensArr


# }}}


@contextmanager
def videoCapture(
    videoSource: Union[int, str],
    cameraResolution: Optional[tuple[int, int]] = None,
    cameraSettings: Optional[str] = None,
    warningLevel=2,
) -> Iterator[cv.VideoCapture]:
    # {{{
    # {{{
    """
    Context manager for working with a cv2.VideoCapture instance.

    Parameters
    ----------
    videoSource : int or str
        The first argument to cv2.VideoCapture. If an int, it will take the
        corresponding detected camera as the video source. If a str, It'll take
        it as a path in the filesystem for a video file.

    cameraResolution : tuple of 2 ints, default=None
        Used to optionally change the camera resolution before handing over the
        VideoCapture instance. If reading from a video file, it does nothing.

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
        This context manager performs no other checks other than those
        indicated on the Raises section. This warrants mentioning because
        OpenCV gives the most cryptic, confusing and even misleading error
        messages, so care must be taken.
    """
    # }}}

    cap: cv.VideoCapture

    if isinstance(videoSource, int):
        if not checkCaptureDevice(videoSource):
            raise ValueError(
                f"The provided capture device index ({videoSource}) is not a "
                "valid capture device. For a list of available capture"
                " devices, use listCaptureDevices"
            )

        cap = CaptureDevice(videoSource, cameraSettings, warningLevel)

        # Set camera resolution
        if cameraResolution:
            resX, resY = cameraResolution
            cap.frameSize = resX, resY

    elif isinstance(videoSource, str):
        if not isfile(videoSource):
            raise ValueError(
                f"The path ({videoSource}) passed to the videoCapture context "
                "manager is not a file."
            )
        cap = cv.VideoCapture(videoSource)

    else:
        raise TypeError(
            f"Invalid type of {type(videoSource)} passed as the first argument"
            "to videoCapture. Valid types: int "
            "(for a video capture device) and"
            "str (for a path to a video in the filesystem."
        )

    try:
        yield cap
    finally:
        cap.release()
        cv.destroyAllWindows()


# }}}


def listCaptureDevices(checkUpTo: int = 10) -> list[int]:
    # {{{
    # {{{
    """
    Checks the first checkUpTo indexes for cv2.VideoCapture and returns a list
    with all the indexes that are available. See checkCaptureDevice.
    """
    # }}}
    capDeviceList: list[int] = []
    for index in range(0, checkUpTo):
        if checkCaptureDevice(index):
            capDeviceList.append(index)
    return capDeviceList


# }}}


def checkCaptureDevice(capDeviceIndex: int) -> bool:
    # {{{
    # {{{
    """
    Checks if the capture device with the provided index is available,
    returning True if it is and False otherwise.
    """
    # }}}
    if not isinstance(capDeviceIndex, int):
        raise TypeError(
            f"Invalid value of {capDeviceIndex} for the capture device index. "
            "Valid values are int."
        )

    cap = cv.VideoCapture(capDeviceIndex)
    if cap.read()[0]:
        cap.release()
        return True
    return False


# }}}


def frameReader(
    capture: cv.VideoCapture,
    rescaleFactor: Real = 1.0,
) -> Generator[Array, None, None]:
    # {{{
    # {{{
    """
    A generator for reading each frame from the VideoCapture instance, and
    optionally
    rescaling it.

    Parameters
    ----------
    capture : cv.VideoCapture
        The OpenCV VideoCapture instance from which to extract the frames. See
        pyCRT.videoReading.videoCapture

    rescaleFactor : int or float, default=1.0
        Factor by which each frame will be scaled. This can help reduce the
        load on the hardware and speed up computation. By default the video
        won't be scaled.

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
) -> Generator[None, Array, None]:
    # {{{
    # {{{
    """
    A generator that acts as a coroutine for recording the capture's frames.
    The idea is that you initialize this generator with the arguments described
    below and send each frame to be recorded with the send method.

    Parameters
    ----------
    recordingPath : str, default=None
        The path (with the extension!) in the filesystem wherein to save the
        recording.

    codecFourcc : str, default='mp4v'
        The fourcc identifier for the video codec to be used for the recording.
        Refer to www.fourcc.org/codecs.php for a list of possible codes.

    recordingFps : int, default=None
        The FPS (frames per second) for the recording, which doesn't need to
        correspond to the FPS of the camera or the source video.
    """
    # }}}

    # This first recieved frame is used to pass the correct frame dimentions to
    # cv.VideoWriter
    frame = yield
    writer = cv.VideoWriter(
        recordingPath,
        cv.VideoWriter.fourcc(*codecFourcc),
        recordingFps,
        frame.shape[1::-1],
    )
    writer.write(frame)

    while True:
        frame = yield
        writer.write(frame)


# }}}


class CaptureDevice(cv.VideoCapture):
    # {{{
    def __init__(
        self,
        videoSource: int,
        cameraSettings: Optional[Union[str, TOMLDict]] = None,
        warningLevel: int = 2,
    ):
        # {{{
        self.raiseExceptions = True
        self.raiseWarnings = True

        if warningLevel == 0:
            self.raiseExceptions = False
            self.raiseWarnings = False
        if warningLevel == 1:
            self.raiseExceptions = False

        if cameraSettings is not None:
            if isinstance(cameraSettings, str):
                self.cameraSettings = loadTOML(cameraSettings)
            elif isinstance(cameraSettings, dict):
                self.cameraSettings = cameraSettings
            else:
                raise TypeError(
                    "Invalid type for cameraSettings. Valid types: "
                    "str (path to a "
                    "TOML settings file) or dict."
                )

        else:
            self.cameraSettings = {}

        self.videoSource = videoSource
        super().__init__(videoSource)

        if self.cameraSettings != {}:
            self.resetValues()

    # }}}

    def get(self, prop: Union[str, int]) -> Union[float, bool]:
        # {{{
        propCode, propName = self.propCodeAndName(prop)

        receivedValue = super().get(propCode)

        if receivedValue == -1:
            self.handleWarnings(
                f"{str(self)} doesn't support the property '{propName}'."
            )

        if propName in BOOLEAN_CODES:
            return self._getBool(prop, receivedValue)

        return receivedValue

    # }}}

    def _getBool(
        self, prop: Union[str, int], receivedValue: float
    ) -> Union[float, bool]:
        # {{{
        _, propName = self.propCodeAndName(prop)

        if propName in self.cameraSettings:
            propTable = self.cameraSettings[propName]

            if isinstance(propTable, dict):
                propValues = propTable["values"]

                if isinstance(propValues, dict):
                    try:
                        onValue = propValues["on"]
                        receivedValue = receivedValue == onValue
                    except KeyError:
                        self.handleWarnings(
                            f"The {propName} property doesn't have an "
                            "'on' or 'off' value declared in the camera's "
                            "specification file."
                        )
            else:
                self.handleWarnings(
                    f"The {propName} property doesn't have an 'on' or "
                    "'off' value declared in the camera's "
                    "specification file."
                )

        return receivedValue

    # }}}

    def set(self, prop: Union[str, int], value: Union[int, bool, float]):
        # {{{
        propCode, propName = self.propCodeAndName(prop)

        if propName in self.cameraSettings and isinstance(value, bool):
            value = self._setBool(prop, value)

        elif isinstance(value, (int, float)):
            if not self.isValid(propName, value):
                self.handleWarnings(
                    "The capture device doesn't support setting "
                    f"'{propName}' to {value}."
                )
        else:
            raise TypeError(
                f"Invalid type ({type(value)}) for the {propName} "
                "property value. Valid types are int, float or bool."
            )

        value = float(value)
        setStatus = super().set(propCode, value)

        if not setStatus:
            self.handleWarnings(
                "The capture device doesn't support setting "
                f"'{propName}' to {value}."
            )

        sleepTime = self.cameraSettings.get("sleepTime", 1)
        if not isinstance(sleepTime, (int, float)):
            raise TypeError(
                f"Invalid type ({type(sleepTime)}) for the sleepTime camera "
                "setting. Valid types are float or int."
            )
        sleep(sleepTime)

    # }}}

    def _setBool(self, prop: Union[str, int], value: bool):
        # {{{
        _, propName = self.propCodeAndName(prop)

        propTable = self.cameraSettings[propName]
        if isinstance(propTable, dict):
            propValues = propTable["values"]
            if isinstance(propValues, dict):
                try:
                    onValue = propValues["on"]
                    offValue = propValues["off"]
                    value = onValue if value else offValue
                    return value
                except KeyError:
                    self.handleWarnings(
                        f"The {propName} property doesn't have an "
                        "'on' or 'off' "
                        "value declared in the camera's"
                        "specification file."
                    )

        self.handleWarnings(
            f"The {propName} property doesn't have an 'on' or "
            " 'off' "
            "value declared in the camera's"
            "specification file."
        )

        return value

    # }}}

    @staticmethod
    def propCodeAndName(prop: Union[str, int]):
        # {{{
        if isinstance(prop, str):
            propCode = ALL_CODES[prop]
            propName = prop
        else:
            propCode = prop
            propName = ALL_PROPS[prop]

        return propCode, propName

    # }}}

    @property
    def frameSize(self) -> tuple[int, int]:
        return (self.width, self.height)

    @frameSize.setter
    def frameSize(self, value: Sequence[int]):
        # {{{
        if len(value) != 2:
            raise ValueError(
                f"{value} is an invalid value for the camera's frame size. "
                "Expected a tuple or list of 2 ints (width and height)."
            )

        width, height = value
        self.width = width
        self.height = height

    # }}}

    def handleWarnings(self, warningStr: str):
        # {{{
        if self.raiseExceptions:
            raise ValueError(warningStr)
        if self.raiseWarnings:
            warn(warningStr, RuntimeWarning)

    # }}}

    def isValid(self, propName: str, value: Union[int, float]) -> bool:
        # {{{
        try:
            propTable = self.cameraSettings[propName]
        except KeyError:
            return True

        if not isinstance(propTable, dict):
            raise TypeError(
                f"Invalid type for the {propName} property. Expected a "
                "dictionary but got a {type(propTable)}"
            )

        propValidValues = propTable["values"]

        if isinstance(propValidValues, list):
            if value in propValidValues:
                return True
        if "min" in propValidValues:
            minVal = propValidValues["min"]
            maxVal = propValidValues["max"]
            step = propValidValues["step"]
            if (value % step == 0) and minVal <= value <= maxVal:
                return True
        elif isinstance(propValidValues, dict):
            if value in propValidValues.values():
                return True
        return False

    # }}}

    def resetValues(self):
        # {{{
        if self.cameraSettings == {}:
            self.handleWarnings(
                "The cameraSettings dict is empty; resetValues will have no "
                "effect."
            )
        for propName, table in self.cameraSettings.items():
            if propName in ALL_CODES or propName == "frameSize":
                if "initial-value" in table:
                    setattr(self, propName, table["initial-value"])
                elif "default" in table:
                    setattr(self, propName, table["default"])

    # }}}


# }}}
