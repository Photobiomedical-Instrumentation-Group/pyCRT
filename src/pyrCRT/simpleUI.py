"""
A simplified object-oriented interface for pyrCRT (pyrCRT.simpleUI)

This module provides the RCRT class, which is meant to be the simplest way to use
pyrCRT's functions distributed among it's other modules.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# pylint: disable=no-name-in-module,import-error
from numpy.typing import NDArray

# pylint: disable=import-error
from arrayOperations import (
    findValueIndex,
    minMaxNormalize,
    sliceByTime,
    sliceFromLocalMax,
    sliceFromMaxToEnd,
    subtractMinimum,
)

# pylint: disable=import-error
from arrayPlotting import (
    saveAvgIntensPlot,
    saveRCRTPlot,
    showAvgIntensPlot,
    showRCRTPlot,
)

# pylint: disable=import-error
from curveFitting import (
    calculateRelativeUncertainty,
    findMaxDivergencePeaks,
    fitExponential,
    fitPolynomial,
    fitRCRT,
    rCRTFromParameters,
)

# pylint: disable=import-error
from videoReading import readVideo

# Type aliases for commonly used types
# {{{
# Array of arbitraty size with float elements.
Array = NDArray[np.float_]

# Tuples of two numpy arrays, typically an array of the timestamp for each frame and an
# array of average intensities within a given ROI
ArrayTuple = tuple[Array, Array]

# Type for something that can be used as the parameters for some curve-fitting function
ParameterSequence = Union[Sequence[float], Array]

# The return type for functions that fit curves. The first element is the optimized
# parameters and the second their standard deviations
FitParametersTuple = tuple[ParameterSequence, ParameterSequence]

Real = Union[float, int, np.float_, np.int_]

# This accounts for the fact that np.int_ doesn't inherit from int
Integer = Union[int, np.int_]

FigAxTuple = tuple[Figure, Axes]

# Standard ROI tuple used by OpenCV
RoiTuple = tuple[int, int, int, int]

# Either a RoiTuple, or "all"
RoiType = Union[RoiTuple, str]
# }}}


# Constants
CHANNEL_INDICES_DICT = {"b": 0, "g": 1, "r": 2}


def calcRCRT(
    timeScds: Array,
    avgIntens: Array,
    criticalTime: Optional[Union[float, list[float]]] = None,
    expTuple: Optional[FitParametersTuple] = None,
    polyTuple: Optional[FitParametersTuple] = None,
    rCRTInitialGuesses: Optional[ParameterSequence] = None,
    exclusionMethod: str = "best fit",
    exclusionCriteria: float = np.inf,
) -> Tuple[FitParametersTuple, float]:
    # {{{
    """cum"""

    if criticalTime is None and (expTuple is None or polyTuple is None):
        maxDivList = findMaxDivergencePeaks(timeScds, avgIntens)

    elif expTuple is not None and polyTuple is not None:
        maxDivList = findMaxDivergencePeaks(
            timeScds, expTuple=expTuple, polyTuple=polyTuple
        )

    elif isinstance(criticalTime, list):
        maxDivList = [findValueIndex(timeScds, x) for x in criticalTime]

    elif isinstance(criticalTime, float):
        maxDiv = findValueIndex(timeScds, criticalTime)
        return calcRCRTStrict(
            timeScds, avgIntens, maxDiv, rCRTInitialGuesses, exclusionCriteria
        )

    if exclusionMethod == "best fit":
        return calcRCRTBestFit(
            timeScds, avgIntens, maxDivList, rCRTInitialGuesses, exclusionCriteria
        )

    if exclusionMethod == "strict":
        return calcRCRTStrict(
            timeScds, avgIntens, maxDivList[0], rCRTInitialGuesses, exclusionCriteria
        )

    if exclusionMethod == "first that works":
        return calcRCRTFirstThatWorks(
            timeScds, avgIntens, maxDivList, rCRTInitialGuesses, exclusionCriteria
        )

    raise ValueError(
        f"Invalid value of {exclusionMethod} passed as exclusionMethod. "
        "Valid values: 'best fit', 'strict' and 'first that works'."
    )


# }}}


def calcRCRTBestFit(
    timeScds: Array,
    avgIntens: Array,
    maxDivList: list[int],
    rCRTInitialGuesses: Optional[ParameterSequence] = None,
    exclusionCriteria: float = np.inf,
) -> Tuple[FitParametersTuple, float]:
    # {{{
    """cum"""

    # A dictionary whose keys are maximum divergence indexes (maxDivs) and values
    # the rCRT and its uncertainty calculated with the respective maxDiv
    maxDivResults: dict[int, FitParametersTuple] = {}
    for maxDiv in maxDivList:
        try:
            rCRTTuple, maxDiv = fitRCRT(
                timeScds,
                avgIntens,
                rCRTInitialGuesses,
                maxDiv,
            )
            maxDivResults[maxDiv] = rCRTTuple
        except RuntimeError:
            pass

    if not maxDivResults:
        raise RuntimeError(
            "rCRT fit failed on all maximum divergence indexes: "
            f"{maxDivList} with initial guesses = {[rCRTInitialGuesses]}"
        )

    maxDiv = min(
        maxDivResults, key=lambda x: calculateRelativeUncertainty(maxDivResults[x])
    )
    rCRTTuple = maxDivResults[maxDiv]

    relativeUncertainty = calculateRelativeUncertainty(maxDivResults[maxDiv])

    if relativeUncertainty > exclusionCriteria:
        raise RuntimeError(
            "Resulting rCRT parameters did not pass the exclusion criteria of "
            f"{exclusionCriteria}. Relative uncertainty: {relativeUncertainty}."
        )

    return rCRTTuple, timeScds[maxDiv]


# }}}


def calcRCRTStrict(
    timeScds: Array,
    avgIntens: Array,
    maxDiv: int,
    rCRTInitialGuesses: Optional[ParameterSequence] = None,
    exclusionCriteria: float = np.inf,
) -> Tuple[FitParametersTuple, float]:
    # {{{
    """cum"""

    rCRTTuple, maxDiv = fitRCRT(
        timeScds,
        avgIntens,
        rCRTInitialGuesses,
        maxDiv,
    )

    relativeUncertainty = calculateRelativeUncertainty(rCRTTuple)

    if relativeUncertainty > exclusionCriteria:
        raise RuntimeError(
            "Resulting rCRT parameters did not pass the exclusion criteria of "
            f"{exclusionCriteria}. Relative uncertainty: {relativeUncertainty}."
        )

    return rCRTTuple, timeScds[maxDiv]


# }}}


def calcRCRTFirstThatWorks(
    timeScds: Array,
    avgIntens: Array,
    maxDivList: Iterable[int],
    rCRTInitialGuesses: Optional[ParameterSequence] = None,
    exclusionCriteria: float = np.inf,
) -> Tuple[FitParametersTuple, float]:
    # {{{
    """cum"""

    maxDivList = sorted(maxDivList)
    for maxDiv in maxDivList:
        try:
            rCRTTuple, maxDiv = fitRCRT(
                timeScds,
                avgIntens,
                rCRTInitialGuesses,
                maxDiv,
            )

            relativeUncertainty = calculateRelativeUncertainty(rCRTTuple)
            if relativeUncertainty < exclusionCriteria:
                return rCRTTuple, timeScds[maxDiv]

        except RuntimeError:
            pass

    raise RuntimeError(
        "rCRT fit failed on all maximum divergence indexes: "
        f"{maxDivList} with initial guesses = {[rCRTInitialGuesses]}"
    )


# }}}


class RCRT:
    def __init__(
        self,
        fullTimeScds: Array,
        fullAvgIntens: Array,
        channelToUse: str = "g",
        fromTime: Optional[Real] = None,
        toTime: Optional[Real] = None,
        funcParams: dict[str, FitParametersTuple] = {},
        sliceMethod: str = "from local max",
        exclusionCriteria: float = 0.12,
        criticalTime: Optional[float] = None,
        initialGuesses: dict[str, ParameterSequence] = {},
        exclusionMethod: str = "first that works",
    ):
        # {{{
        """cum"""

        self.fullTimeScds = fullTimeScds
        self.fullAvgIntens = fullAvgIntens
        self.usedChannel = channelToUse.strip().lower()
        self.setSlice(fromTime, toTime, sliceMethod)

        self.initialGuesses = initialGuesses

        if "exponential" in funcParams:
            self.expParams, self.expStdDev = funcParams["exponential"]
        else:
            self.expParams, self.expStdDev = fitExponential(
                self.timeScds,
                self.avgIntens,
                self.initialGuesses.get("exponential", None),
            )

        if "polynomial" in funcParams:
            self.polyParams, self.polyStdDev = funcParams["polynomial"]
        else:
            self.polyParams, self.polyStdDev = fitPolynomial(
                self.timeScds,
                self.avgIntens,
                self.initialGuesses.get("polynomial", None),
            )

        if "rCRT" in funcParams and criticalTime is not None:
            self.rCRTParams, self.rCRTStdDev = funcParams["rCRT"]
            self.criticalTime = criticalTime
        else:
            (self.rCRTParams, self.rCRTStdDev), self.criticalTime = self.calcRCRT(
                criticalTime,
                self.initialGuesses.get("rCRT", None),
                exclusionMethod,
                exclusionCriteria,
            )

    # }}}

    def calcRCRT(
        self,
        criticalTime: Optional[float] = None,
        rCRTInitialGuesses: Optional[ParameterSequence] = None,
        exclusionMethod: str = "best fit",
        exclusionCriteria: float = np.inf,
    ) -> Tuple[FitParametersTuple, float]:
        # {{{
        """cum"""

        return calcRCRT(
            self.timeScds,
            self.avgIntens,
            criticalTime,
            self.expTuple,
            self.polyTuple,
            rCRTInitialGuesses,
            exclusionMethod,
            exclusionCriteria,
        )

    # }}}

    @property
    def B(self) -> Array:
        # {{{
        """cum"""

        return self.fullAvgIntens[:, 0]

    # }}}

    @property
    def G(self) -> Array:
        # {{{
        """cum"""

        return self.fullAvgIntens[:, 1]

    # }}}

    @property
    def R(self) -> Array:
        # {{{
        """cum"""

        return self.fullAvgIntens[:, 2]

    # }}}

    @property
    def channelFullAvgIntens(self) -> Array:
        # {{{
        """cum"""

        return self.fullAvgIntens[:, CHANNEL_INDICES_DICT[self.usedChannel]]

    # }}}

    @property
    def expTuple(self) -> FitParametersTuple:
        # {{{
        """cum"""

        return (self.expParams, self.expStdDev)

    # }}}

    @property
    def polyTuple(self) -> FitParametersTuple:
        # {{{
        """cum"""

        return (self.polyParams, self.polyStdDev)

    # }}}

    @property
    def rCRTTuple(self) -> FitParametersTuple:
        # {{{
        """cum"""

        return (self.rCRTParams, self.rCRTStdDev)

    # }}}

    @property
    def rCRT(self) -> Tuple[float, float]:
        # {{{
        """cum"""

        return rCRTFromParameters(self.rCRTTuple)

    # }}}

    @property
    def criticalTime(self) -> float:
        # {{{
        """cum"""

        self.maxDiv: int
        return self.timeScds[self.maxDiv]

    # }}}

    @criticalTime.setter
    def criticalTime(self, value: Real) -> None:
        # {{{
        """cum"""

        self.maxDiv = int(findValueIndex(self.timeScds, value))

    # }}}

    @property
    def relativeUncertainty(self) -> np.float_:
        # {{{
        """cum"""

        return calculateRelativeUncertainty(self.rCRTTuple)

    # }}}

    def __str__(self) -> str:
        # {{{
        return f"{self.rCRT[0]:.2f}±{100*self.relativeUncertainty:.2f}%"

    # }}}

    def setSlice(
        self,
        fromTime: Optional[Real] = None,
        toTime: Optional[Real] = None,
        sliceMethod: str = "from local max",
    ) -> None:
        # {{{
        """cum"""

        sliceMethod = sliceMethod.strip().lower()

        if sliceMethod == "from max" or (fromTime, toTime) == (None, None):
            self.slice = sliceFromMaxToEnd(self.channelFullAvgIntens)
        elif sliceMethod == "from local max":
            self.slice = sliceFromLocalMax(
                self.fullTimeScds, self.channelFullAvgIntens, fromTime, toTime
            )
        elif sliceMethod == "by time":
            self.slice = sliceByTime(self.fullTimeScds, fromTime, toTime)
        else:
            raise ValueError(
                f"Invalid value of '{sliceMethod}' passed as sliceMethod. "
                "Valid values: 'from local max', 'from max' and 'by time'"
            )

        fromIndex, toIndex, _ = self.slice.indices(len(self.fullTimeScds))
        self.fromTime, self.toTime = (
            self.fullTimeScds[fromIndex],
            self.fullTimeScds[toIndex],
        )
        self.timeScds = subtractMinimum(self.fullTimeScds[self.slice])
        self.avgIntens = minMaxNormalize(self.channelFullAvgIntens[self.slice])

    # }}}

    def showAvgIntensPlot(self) -> None:
        # {{{
        """cum"""

        showAvgIntensPlot(self.fullTimeScds, self.fullAvgIntens)

    # }}}

    def saveAvgIntensPlot(self, figPath: str) -> None:
        # {{{
        """cum"""

        saveAvgIntensPlot(figPath, self.fullTimeScds, self.fullAvgIntens)

    # }}}

    def showRCRTPlot(self) -> None:
        # {{{
        """cum"""

        showRCRTPlot(
            self.timeScds,
            self.avgIntens,
            {
                "exponential": self.expTuple,
                "polynomial": self.polyTuple,
                "rCRT": self.rCRTTuple,
            },
            self.criticalTime,
            self.usedChannel,
        )

    # }}}

    def saveRCRTPlot(self, figPath: str) -> None:
        # {{{
        """cum"""

        saveRCRTPlot(
            figPath,
            self.timeScds,
            self.avgIntens,
            {
                "exponential": self.expTuple,
                "polynomial": self.polyTuple,
                "rCRT": self.rCRTTuple,
            },
            self.criticalTime,
            self.usedChannel,
        )

    # }}}

    @classmethod
    def fromVideoFile(
        cls,
        videoPath: str,
        roi: Optional[RoiType] = None,
        displayVideo: bool = True,
        rescaleFactor: Real = 1.0,
        waitKeyTime: int = 1,
        **kwargs: Any,
    ) -> RCRT:
        # {{{
        """cum"""

        fullTimeScds, fullAvgIntens = readVideo(
            videoPath,
            roi=roi,
            displayVideo=displayVideo,
            rescaleFactor=rescaleFactor,
            waitKeyTime=waitKeyTime,
        )
        return cls(fullTimeScds, fullAvgIntens, **kwargs)

    # }}}

    @classmethod
    def fromCaptureDevice(
        cls,
        videoSource: int,
        roi: Optional[RoiType] = None,
        cameraResolution: Optional[Tuple[int, int]] = None,
        recordingPath: Optional[str] = None,
        codecFourcc: str = "mp4v",
        recordingFps: float = 30.0,
        **kwargs: Any,
    ) -> RCRT:
        # {{{
        """cum"""

        fullTimeScds, fullAvgIntens = readVideo(
            videoSource,
            roi=roi,
            cameraResolution=cameraResolution,
            recordingPath=recordingPath,
            codecFourcc=codecFourcc,
            recordingFps=recordingFps,
        )
        return cls(fullTimeScds, fullAvgIntens, **kwargs)

    # }}}

    @classmethod
    def fromArchive(cls, filePath: str) -> RCRT:
        # {{{
        """cum"""

        archive = np.load(filePath)

        return cls(
            fullTimeScds=archive["fullTimeScds"],
            fullAvgIntens=archive["fullAvgIntens"],
            channelToUse=str(archive["channelToUse"]),
            fromTime=float(archive["fromTime"]),
            toTime=float(archive["toTime"]),
            funcParams={
                "exponential": tuple(archive["expTuple"]),  # type: ignore
                "polynomial": tuple(archive["polyTuple"]),  # type: ignore
                "rCRT": tuple(archive["rCRTTuple"]),  # type: ignore
            },
            criticalTime=float(archive["criticalTime"]),
            sliceMethod="from local max",
        )
        # }}}

    def save(self, filePath: str) -> None:
        # {{{
        """cum"""

        np.savez(
            filePath,
            fullTimeScds=self.fullTimeScds,
            fullAvgIntens=self.fullAvgIntens,
            channelToUse=self.usedChannel,
            fromTime=self.fromTime,
            toTime=self.toTime,
            expTuple=np.array(self.expTuple),
            polyTuple=np.array(self.polyTuple),
            rCRTTuple=np.array(self.rCRTTuple),
            criticalTime=self.criticalTime,
        )


# }}}
