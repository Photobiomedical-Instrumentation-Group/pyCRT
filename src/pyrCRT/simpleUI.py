"""
A simplified object-oriented interface for pyrCRT (pyrCRT.simpleUI)

This module provides the RCRT class, which is meant to be the simplest way to use
pyrCRT's functions distributed among it's other modules.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union, overload

import numpy as np

# pylint: disable=import-error
from arrayOperations import (
    minMaxNormalize,
    sliceByTime,
    sliceFromLocalMax,
    sliceFromMaxToEnd,
)

# pylint: disable=import-error
from arrayPlotting import showAvgIntens, showAvgIntensAndFunctions

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
# Used just as a shorthand
Array = np.ndarray

# Tuples of two numpy arrays, typically an array of the timestamp for each frame and an
# array of average intensities within a given ROI
ArrayTuple = Tuple[Array, Array]

# Tuple of two lists, the first being the fitted parameters and the second their
# standard deviations
FitParametersTuple = Tuple[Array, Array]

Real = Union[float, int, np.float_, np.int_]

# This accounts for the fact that np.int_ doesn't inherit from int
Integer = Union[int, np.int_]
# }}}


# Constants
CHANNEL_INDICES_DICT = {"b": 0, "g": 1, "r": 2}


class RCRT:
    def __init__(
        self,
        fullTimeScds: Array,
        fullAvgIntens: Array,
        channelToUse: str = "g",
        fromTime: Optional[Real] = None,
        toTime: Optional[Real] = None,
        funcParams: Dict[str, FitParametersTuple] = {},
        sliceMethod: str = "from local max",
        exclusionCriteria: float = 0.12,
        criticalTime: Optional[float] = None,
        initialGuesses: Dict[str, List[Real]] = {},
        exclusionMethod: str = "best fit",
        **kwargs: Any,
    ):
        # {{{
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
            self.maxDiv = self.criticalTimeToMaxDiv(criticalTime)
        else:
            (self.rCRTParams, self.rCRTStdDev), self.maxDiv = self.calcRCRT(
                criticalTime,
                self.initialGuesses.get("rCRT", None),
                exclusionMethod,
                exclusionCriteria,
            )

    # }}}

    def calcRCRT(
        self,
        criticalTime: Optional[float] = None,
        rCRTInitialGuesses: Optional[List[Real]] = None,
        exclusionMethod: str = "best fit",
        exclusionCriteria: float = np.inf,
    ) -> Tuple[FitParametersTuple, int]:
        # {{{

        if criticalTime is None:
            maxDivPeaks = findMaxDivergencePeaks(
                self.timeScds, expTuple=self.expTuple, polyTuple=self.polyTuple
            )

            if exclusionMethod == "best fit":
                return self.calcRCRTBestFit(
                    maxDivPeaks, rCRTInitialGuesses, exclusionCriteria
                )

            if exclusionMethod == "strict":
                return self.calcRCRTStrict(
                    maxDivPeaks[0], rCRTInitialGuesses, exclusionCriteria
                )

            if exclusionMethod == "first that works":
                return self.calcRCRTBestFit(
                    maxDivPeaks, rCRTInitialGuesses, exclusionCriteria
                )

            raise ValueError(
                f"Invalid value of {exclusionMethod} passed as exclusionMethod. "
                "Valid values: 'best fit', 'strict' and 'first that works'."
            )

        maxDiv = self.criticalTimeToMaxDiv(criticalTime)

        return self.calcRCRTStrict(
            maxDivPeaks[0], rCRTInitialGuesses, exclusionCriteria
        )

    # }}}

    def calcRCRTBestFit(
        self,
        maxDivList: List[int],
        rCRTInitialGuesses: Optional[List[Real]] = None,
        exclusionCriteria: float = np.inf,
    ) -> Tuple[FitParametersTuple, int]:
        # {{{
        # A dictionary whose keys are maximum divergence indexes (maxDivs) and values
        # the rCRT and its uncertainty calculated with the respective maxDiv
        maxDivResults: Dict[int, FitParametersTuple] = {}
        for maxDiv in maxDivList:
            try:
                rCRTTuple, maxDiv = fitRCRT(
                    self.timeScds,
                    self.avgIntens,
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

        relativeUncertainty = calculateRelativeUncertainty(maxDivResults[maxDiv])

        if relativeUncertainty > exclusionCriteria:
            raise RuntimeError(
                f"Resulting rCRT parameters did not pass the exclusion criteria of "
                "{exclusionCriteria}. Relative uncertainty: {relativeUncertainty}."
            )

        return rCRTTuple, maxDiv

    # }}}

    def calcRCRTStrict(
        self,
        maxDiv: int,
        rCRTInitialGuesses: Optional[List[Real]] = None,
        exclusionCriteria: float = np.inf,
    ) -> Tuple[FitParametersTuple, int]:
        # {{{
        rCRTTuple, maxDiv = fitRCRT(
            self.timeScds,
            self.avgIntens,
            rCRTInitialGuesses,
            maxDiv,
        )

        relativeUncertainty = calculateRelativeUncertainty(rCRTTuple)

        if relativeUncertainty > exclusionCriteria:
            raise RuntimeError(
                "Resulting rCRT parameters did not pass the exclusion criteria of "
                f"{exclusionCriteria}. Relative uncertainty: {relativeUncertainty}."
            )

        return rCRTTuple, maxDiv

    # }}}

    def calcRCRTFirstThatWorks(
        self,
        maxDivList: List[int],
        rCRTInitialGuesses: Optional[List[Real]] = None,
        exclusionCriteria: float = np.inf,
    ) -> Tuple[FitParametersTuple, int]:
        # {{{

        for maxDiv in maxDivList:
            try:
                rCRTTuple, maxDiv = fitRCRT(
                    self.timeScds,
                    self.avgIntens,
                    rCRTInitialGuesses,
                    maxDiv,
                )

                relativeUncertainty = calculateRelativeUncertainty(rCRTTuple)
                if relativeUncertainty > exclusionCriteria:
                    return rCRTTuple, maxDiv

            except RuntimeError:
                pass

        raise RuntimeError(
            f"rCRT fit failed on all maximum divergence indexes: "
            "{maxDivList} with initial guesses = {[rCRTInitialGuesses]}"
        )

    # }}}

    @property
    def B(self) -> Array:
        return self.fullAvgIntens[:, 0]

    @property
    def G(self) -> Array:
        return self.fullAvgIntens[:, 1]

    @property
    def R(self) -> Array:
        return self.fullAvgIntens[:, 2]

    @property
    def channelFullAvgIntens(self) -> Array:
        return self.fullAvgIntens[:, CHANNEL_INDICES_DICT[self.usedChannel]]

    @property
    def expTuple(self) -> FitParametersTuple:
        return (self.expParams, self.expStdDev)

    @property
    def polyTuple(self) -> FitParametersTuple:
        return (self.polyParams, self.polyStdDev)

    @property
    def rCRTTuple(self) -> FitParametersTuple:
        return (self.rCRTParams, self.rCRTStdDev)

    @property
    def rCRT(self) -> Tuple[np.float_, np.float_]:
        return rCRTFromParameters(self.rCRTTuple)

    @property
    def criticalTime(self) -> np.float_:
        return self.timeScds[self.maxDiv]

    @property
    def relativeUncertainty(self) -> np.float_:
        return calculateRelativeUncertainty(self.rCRTTuple)

    def criticalTimeToMaxDiv(self, criticalTime: float) -> int:
        return int(np.where(self.timeScds >= criticalTime)[0][0])

    def setSlice(
        self,
        fromTime: Optional[Real] = None,
        toTime: Optional[Real] = None,
        sliceMethod: str = "from local max",
    ) -> None:
        # {{{
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

        self.timeScds = self.fullTimeScds[self.slice]
        self.avgIntens = minMaxNormalize(self.channelFullAvgIntens[self.slice])

    # }}}

    def showFullAvgIntens(
        self,
    ) -> None:
        # {{{
        showAvgIntens(self.fullTimeScds, self.fullAvgIntens, "bgr")

    # }}}

    def showRCRTPlot(
        self,
    ) -> None:
        # {{{
        showAvgIntensAndFunctions(
            self.timeScds,
            self.avgIntens,
            funcParams={
                "exponential": self.expParams,
                "polynomial": self.polyParams,
                "rCRT": self.rCRTParams,
            },
            maxDiv=self.maxDiv,
            funcOptions={"intensities": {"channels": self.usedChannel}}
        )

    # }}}

    @classmethod
    def fromVideoFile(
        cls,
        videoPath: str,
        **kwargs: Any,
    ) -> RCRT:
        # {{{
        fullTimeScds, fullAvgIntens = readVideo(videoPath, **kwargs)
        return cls(fullTimeScds, fullAvgIntens, **kwargs)

    # }}}

    @classmethod
    def fromArchive(cls, filePath: str) -> RCRT:
        # {{{
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
        np.savez(
            filePath,
            fullTimeScds=self.fullTimeScds,
            fullAvgIntens=self.fullAvgIntens,
            channelToUse=self.usedChannel,
            fromTime=self.timeScds[0],
            toTime=self.timeScds[-1],
            expTuple=self.expTuple,
            polyTuple=self.polyTuple,
            rCRTTuple=self.rCRTTuple,
            criticalTime=self.criticalTime,
        )


# }}}
