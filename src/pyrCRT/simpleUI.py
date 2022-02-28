"""
A simplified object-oriented interface for pyrCRT (pyrCRT.simpleUI)

This module provides the RCRT class, which is meant to be the simplest way to use
pyrCRT's functions distributed among it's other modules.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# pylint: disable=import-error
from arrayOperations import (
    minMaxNormalize,
    sliceByTime,
    sliceFromLocalMax,
    sliceFromMaxToEnd,
    subtractMinimum,
)

# pylint: disable=import-error
from arrayPlotting import (
    _plotAvgIntensAndFunctions,
    addTextToLabel,
    makeFigAxes,
    plotAvgIntens,
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

FigAxTuple = Tuple[Figure, Axes]
# }}}


# Constants
CHANNEL_INDICES_DICT = {"b": 0, "g": 1, "r": 2}


def figVisualizationFactory(
    func: Callable[[], FigAxTuple]
) -> Tuple[Callable[[], None], Callable[[], None]]:
    # {{{
    def showPlot(self, *args: Any, **kwargs: Any) -> None:
        try:
            fig, _ = func(self, *args, **kwargs)
        except Exception as err:
            raise err
        finally:
            if not plt.isinteractive():
                plt.show()
                plt.close(fig)

    def saveFig(self, figPath: str, *args: Any, **kwargs: Any) -> None:
        try:
            fig, _ = func(self, *args, **kwargs)
            plt.savefig(figPath)
        except Exception as err:
            raise err
        finally:
            if not plt.isinteractive():
                plt.close(fig)

    return showPlot, saveFig

    # }}}


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
        exclusionMethod: str = "first that works",
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
                return self.calcRCRTFirstThatWorks(
                    maxDivPeaks, rCRTInitialGuesses, exclusionCriteria
                )

            raise ValueError(
                f"Invalid value of {exclusionMethod} passed as exclusionMethod. "
                "Valid values: 'best fit', 'strict' and 'first that works'."
            )

        maxDiv = self.criticalTimeToMaxDiv(criticalTime)

        return self.calcRCRTStrict(maxDiv, rCRTInitialGuesses, exclusionCriteria)

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
        rCRTTuple = maxDivResults[maxDiv]

        relativeUncertainty = calculateRelativeUncertainty(maxDivResults[maxDiv])

        if relativeUncertainty > exclusionCriteria:
            raise RuntimeError(
                "Resulting rCRT parameters did not pass the exclusion criteria of "
                f"{exclusionCriteria}. Relative uncertainty: {relativeUncertainty}."
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

        maxDivList = sorted(maxDivList)
        for maxDiv in maxDivList:
            try:
                rCRTTuple, maxDiv = fitRCRT(
                    self.timeScds,
                    self.avgIntens,
                    rCRTInitialGuesses,
                    maxDiv,
                )

                relativeUncertainty = calculateRelativeUncertainty(rCRTTuple)
                if relativeUncertainty < exclusionCriteria:
                    return rCRTTuple, maxDiv

            except RuntimeError:
                pass

        raise RuntimeError(
            "rCRT fit failed on all maximum divergence indexes: "
            f"{maxDivList} with initial guesses = {[rCRTInitialGuesses]}"
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

    def __str__(self) -> str:
        # {{{
        return f"{self.rCRT[0]:.2f}±{100*self.relativeUncertainty:.2f}%"

    # }}}

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

        fromIndex, toIndex, cum = self.slice.indices(len(self.fullTimeScds))
        self.fromTime, self.toTime = (
            self.fullTimeScds[fromIndex],
            self.fullTimeScds[toIndex],
        )
        self.timeScds = subtractMinimum(self.fullTimeScds[self.slice])
        self.avgIntens = minMaxNormalize(self.channelFullAvgIntens[self.slice])

    # }}}

    def makeFullAvgIntensPlot(self) -> FigAxTuple:
        # {{{
        fig, ax = makeFigAxes(
            ("Time (s)", "Average intensities (u.a.)"),
            "Channel average intensities",
        )

        plotAvgIntens(
            (fig, ax),
            self.fullTimeScds,
            self.fullAvgIntens,
        )

        return fig, ax

    # }}}

    def makeRCRTPlot(self) -> FigAxTuple:
        # {{{
        fig, ax = makeFigAxes(
            ("Time since release of compression (s)", "Average intensities (u.a.)"),
            "Average intensities and fitted functions",
        )

        _plotAvgIntensAndFunctions(
            (fig, ax),
            self.timeScds,
            self.avgIntens,
            funcParams={
                "exponential": self.expParams,
                "polynomial": self.polyParams,
                "rCRT": self.rCRTParams,
            },
            maxDiv=self.maxDiv,
            funcOptions={"intensities": {"channels": self.usedChannel}},
        )

        addTextToLabel(ax, f"rCRT={self.__str__()}", loc="upper right")

        return fig, ax

    # }}}

    showFullAvgIntens, saveFullAvgIntens = figVisualizationFactory(
        makeFullAvgIntensPlot
    )

    showRCRTPlot, saveRCRTPlot = figVisualizationFactory(makeRCRTPlot)

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
            fromTime=self.fromTime,
            toTime=self.toTime,
            expTuple=self.expTuple,
            polyTuple=self.polyTuple,
            rCRTTuple=self.rCRTTuple,
            criticalTime=self.criticalTime,
        )


# }}}
