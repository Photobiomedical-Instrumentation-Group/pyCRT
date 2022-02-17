"""
Functions related to curve fitting (pyrCRT.curveFitting)

This module implements the operations necessary to calculate the rCRT from the average
intensities array and the frame times array, namely fitting a polynomial and two
exponential curves on the data.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from curveFitting import exponential, polynomial

# Type aliases for commonly used types
# {{{
# Tuples of two numpy arrays, typically an array of the timestamp for each frame and an
# array of average intensities within a given ROI
ArrayTuple = Tuple[np.ndarray, np.ndarray]

# Tuple of two lists, the first being the fitted parameters and the second their
# standard deviations
FitParametersTuple = Tuple[np.ndarray, np.ndarray]

Real = Union[float, int, np.float_, np.int_]

# This accounts for the fact that np.int_ doesn't inherit from int
Integer = Union[int, np.int_]

FigAxTuple = Tuple[Figure, Axes]
# }}}

# Constants
CHANNEL_INDICES_DICT = {"b": 0, "g": 1, "r": 2}


def setFigSizePx(fig: Figure, figSizePx: Tuple[int, int]) -> None:
    # {{{
    pixels = 1 / fig.get_dpi()
    fig.set_size_inches(dim * pixels for dim in figSizePx)


# }}}


def plotAvgIntens(
    figAxTuple: FigAxTuple,
    timeScdsArr: np.ndarray,
    avgIntenArr: np.ndarray,
    channelsToUse: str = "bgr",
    **kwargs: Any,
) -> None:
    # {{{
    _, ax = figAxTuple

    plotOptions = kwargs.get("plotOptions", None)
    legendOptions = kwargs.get("legendOptions", None)
    if plotOptions is None:
        plotOptions = {}
    if legendOptions is None:
        legendOptions = {}

    for channel in channelsToUse.strip().lower():
        ax.plot(
            timeScdsArr,
            avgIntenArr[:, CHANNEL_INDICES_DICT[channel]],
            color=channel,
            label=f"Channel {channel.upper()}",
            **plotOptions,
        )
    ax.legend(**legendOptions)


# }}}


def plotFunction(
    figAxTuple: FigAxTuple,
    timeScdsArr: np.ndarray,
    func: Callable,
    funcParams: List[Real],
    **kwargs: Any,
) -> None:
    # {{{
    _, ax = figAxTuple

    funcY = func(timeScdsArr, *funcParams)

    plotOptions = kwargs.get("plotOptions", None)
    legendOptions = kwargs.get("legendOptions", None)
    if plotOptions is None:
        plotOptions = {"label": f"{func.__name__}", "ls": "--"}
    if legendOptions is None:
        legendOptions = {}
    ax.plot(timeScdsArr, funcY, **plotOptions)
    ax.legend(**legendOptions)


# }}}


def plotRCRT(
    figAxTuple: FigAxTuple,
    timeScdsArr: np.ndarray,
    rCRTParams: List[Real],
    maxDiv: Optional[Integer] = None,
    **kwargs: Any,
) -> None:
    # {{{
    _, ax = figAxTuple

    funcY = exponential(timeScdsArr, *rCRTParams)

    plotOptions = kwargs.get("plotOptions", None)
    legendOptions = kwargs.get("legendOptions", None)
    if plotOptions is None:
        plotOptions = {"label": "rCRT Exp", "fmt": "c-"}
    if legendOptions is None:
        legendOptions = {}
    ax.plot(timeScdsArr, funcY, **plotOptions)

    if maxDiv is not None:
        ax.axvline(timeScdsArr[maxDiv], label="maxDiv", c="k", ls=":")

    ax.legend(**legendOptions)


# }}}


# This is a filthy hack
def plotAvgIntensAndFunctions(
    figAxTuple: FigAxTuple,
    timeScdsArr: np.ndarray,
    avgIntenArr: np.ndarray,
    funcParams: Dict[str, List[Any]],
    maxDiv: Optional[Integer] = None,
    funcOptions: Optional[Dict[str, Any]] = None,
) -> None:
    # {{{
    # {{{

    # }}}

    if funcOptions is None:
        funcOptions = {}

    plotFunction(
        figAxTuple,
        timeScdsArr,
        exponential,
        funcParams["exponential"],
        **funcOptions.get("exponential", {}),
    )
    plotFunction(
        figAxTuple,
        timeScdsArr,
        polynomial,
        funcParams["polynomial"],
        **funcOptions.get("polynomial", {}),
    )
    plotRCRT(
        figAxTuple,
        timeScdsArr,
        funcParams["rCRT"],
        maxDiv,
        **funcOptions.get("rCRT", {}),
    )
    plotAvgIntens(
        figAxTuple,
        timeScdsArr,
        avgIntenArr,
        **funcOptions.get("intensities", {}),
    )


# }}}


def figVisualizationFactory(
    func: Callable,
    figTitle: Optional[str] = None,
    axisLabels: Tuple[str, str] = ("Time (s)", "Average Intensities (a.u.)"),
    figSizePx: Tuple[int, int] = (960, 600),
    figAxTuple: Optional[FigAxTuple] = None,
    *args: Any,
    **kwargs: Any,
) -> Tuple[Callable, Callable]:
    # {{{

    xlabel, ylabel = axisLabels

    if figAxTuple is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figAxTuple

    setFigSizePx(fig, figSizePx)

    func((fig, ax), *args, **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.autoscale(enable=True, axis="x", tight=True)

    fig.suptitle(figTitle)
    fig.set_tight_layout(True)

    def showPlot() -> None:
        fig.show()

    def saveFig(figPath: str) -> None:
        fig.save(figPath)

    return showPlot, saveFig


# }}}


showAvgIntens, saveAvgIntens = figVisualizationFactory(
    plotAvgIntens, "Channel average intensities"
)

showAvgIntensAndFunctions, saveAvgIntensAndFunctions = figVisualizationFactory(
    plotAvgIntensAndFunctions, "Average Intensities and fitted functions"
)
