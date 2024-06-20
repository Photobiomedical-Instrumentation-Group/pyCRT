"""
Functions related to curve fitting (pyCRT.curveFitting)

This module implements the operations necessary to calculate the pCRT from the
average intensities array and the frame times array, namely fitting a
polynomial and two exponential curves on the data.
"""

from typing import Any, Callable, Generator, Optional, Sequence, Union

import numpy as np
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import AutoLocator, AutoMinorLocator
from numpy.typing import NDArray

from .arrayOperations import minMaxNormalize
from .curveFitting import (
    calculateRelativeUncertainty,
    exponential,
    pCRTFromParameters,
    polynomial,
)

# Type aliases for commonly used types
# {{{
# Array of arbitraty size with float elements.
Array = NDArray[np.float_]

# Tuples of two numpy arrays, typically an array of the timestamp for each
# frame and an array of average intensities within a given ROI
ArrayTuple = tuple[Array, Array]

# Type for something that can be used as the parameters for some curve-fitting
# function
ParameterSequence = Union[Sequence[float], Array]

# The return type for functions that fit curves. The first element is the
# optimized parameters and the second their standard deviations
FitParametersTuple = tuple[ParameterSequence, ParameterSequence]

Real = Union[float, int, np.float_, np.int_]

# This accounts for the fact that np.int_ doesn't inherit from int
Integer = Union[int, np.int_]

FigAxTuple = tuple[Figure, Axes]
# }}}

# Constants
BGR_INDICES_DICT = {"b": 0, "g": 1, "r": 2}
channelColors = {
    "b": "blue",
    "g": "green",
    "r": "red",
    "h": "magenta",
    "l": "gray",
    "s": "orange",
    "v": "cyan",
    "a": "y",
}


class ColorCounter:
    # {{{
    i = 0
    channelColorsList = list(channelColors.values())

    def get(self, key):
        return channelColors[key]
        # try:
        #     return channelColors[key]
        # except KeyError:
        #     channelColors[key] = self.channelColorsList[self.i]
        #     self.i += 1
        #     return channelColors[key]


# }}}

colorCounter = ColorCounter()


def plotAvgIntens(
    figAxTuple: FigAxTuple,
    timeScdsArr: Array,
    channelsAvgIntensArr: Array,
    channels: Optional[str] = None,
    channelNames=BGR_INDICES_DICT,
    normIntens=False,
    **kwargs: Any,
) -> None:
    # {{{
    # {{{
    """
    Plots the average intensities array on the given mpl.Figure and mpl.Axes
    tuple. This can plot a set of channels or all three, depending on
    channelsAvgIntensArr's
    dimentions and the channels argument.

    Parameters
    ----------
    figAxTuple : tuple of mpl.Figure and mpl.Axes respectively
        The figure and axes on which to plot. In practice this function only
        utilizes
        the Axes instance, not the Figure.

    timeScdsArr, channelsAvgIntensArr : np.ndarray
        The arrays of seconds and average intensities corresponding to each
        frame, respectively. This is typically the output of
        videoReading.readVideo.

    channels : str or None, default=None
        Which channels to plot, and also used to set the color of each line in
        the plot and its label. Should be a string with some combination of
        "r", "g" and "b". If it is None and channelsAvgIntensArr is
        one-dimensional, then it'll plot that channel in gray, and if
        channelsAvgIntensArr is not one-dimensional, then channels will be set
        as "bgr".

    **kwargs: dict of Any
        The keyword arguments that may be passed to mpl.Axes.plot.
    """
    # }}}

    _, ax = figAxTuple

    plotOptions = kwargs.get("plotOptions", None)
    legendOptions = kwargs.get("legendOptions", None)
    if plotOptions is None:
        plotOptions = {}
    if legendOptions is None:
        legendOptions = {}

    if channels is None:
        if len(channelsAvgIntensArr.shape) == 1:  # a single channel
            if normIntens:
                channelsAvgIntensArr = minMaxNormalize(channelsAvgIntensArr)
            ax.plot(
                timeScdsArr,
                channelsAvgIntensArr,
                color="grey",
                label="Channel",
                **plotOptions,
            )
        else:
            channels = "".join(list(channelNames.keys()))
            plotAvgIntens(
                figAxTuple,
                timeScdsArr,
                channelsAvgIntensArr,
                channels=channels,
                channelNames=channelNames,
                normIntens=normIntens,
            )
    else:
        channels = channels.strip().lower()
        if len(channels) == 1:
            channelIntens = channelsAvgIntensArr
            if normIntens:
                channelIntens = minMaxNormalize(channelIntens)

            ax.plot(
                timeScdsArr,
                channelIntens,
                color=colorCounter.get(channels),
                label=f"Channel {channels.upper()}",
                **plotOptions,
            )
        else:
            for channel in channels.strip().lower():
                channelIntens = channelsAvgIntensArr[:, channelNames[channel]]
                if normIntens:
                    channelIntens = minMaxNormalize(channelIntens)

                ax.plot(
                    timeScdsArr,
                    channelIntens,
                    color=colorCounter.get(channel),
                    label=f"Channel {channel.upper()}",
                    **plotOptions,
                )
    ax.legend(**legendOptions)


# }}}


def liveAvgIntensPlot(
    numPoints: int = 199, figSizePx: tuple[int, int] = (480, 300)
) -> Generator[None, ArrayTuple, None]:
    # {{{
    fig, ax = makeFigAxes(
        ("Time (normalized)", "Average Intensities (a.u.)"),
        figSizePx=figSizePx,
    )

    timeScdsArr = np.linspace(0.0, 1.0, numPoints)
    channelsAvgIntensArr = np.array([[0, 0, 0]] * numPoints)

    bLine = ax.plot(
        timeScdsArr, channelsAvgIntensArr[:, 0], "b", label="Channel 0"
    )[0]
    gLine = ax.plot(
        timeScdsArr, channelsAvgIntensArr[:, 1], "g", label="Channel 1"
    )[0]
    rLine = ax.plot(
        timeScdsArr, channelsAvgIntensArr[:, 2], "r", label="Channel 2"
    )[0]
    ax.set_ylim(0, 255)
    ax.legend()

    fig.canvas.draw()
    axbackground = fig.canvas.copy_from_bbox(ax.bbox)
    plt.show(block=False)

    while True:
        _, channelsAvgIntens = yield

        channelsAvgIntensArr[1:] = channelsAvgIntensArr[:-1]
        channelsAvgIntensArr[0] = channelsAvgIntens
        bLine.set_data(timeScdsArr, channelsAvgIntensArr[:, 0])
        gLine.set_data(timeScdsArr, channelsAvgIntensArr[:, 1])
        rLine.set_data(timeScdsArr, channelsAvgIntensArr[:, 2])

        fig.canvas.restore_region(axbackground)
        ax.draw_artist(bLine)
        ax.draw_artist(gLine)
        ax.draw_artist(rLine)
        fig.canvas.blit(ax.bbox)
        fig.canvas.flush_events()


# }}}


def plotFunction(
    figAxTuple: FigAxTuple,
    timeScdsArr: Array,
    func: Callable[..., Array],
    funcParams: ParameterSequence,
    **kwargs: Any,
) -> None:
    # {{{
    # {{{
    """
    Applies the given function with the given parameters over timeScdsArr and
    plots the resulting array as the Y axis.

    Parameters
    ----------
    figAxTuple : tuple of mpl.Figure and mpl.Axes respectively
        The figure and axes on which to plot. In practice this function only
        utilizes the Axes instance, not the Figure.

    timeScdsArr : np.ndarray
        The array of seconds corresponding to each frame. This is typically the
        first output of videoReading.readVideo

    func : callable that returns a np.ndarray
        The function that will be applied to timeScdsArr with funcParams as its
        parameters. This is typically the polynomial or exponential functions
        from curveFitting.

    funcParams : sequence of float
        This sequence will be unpacked and used as func's positional arguments.
        Again, see curveFitting.exponential or curveFitting.polynomial for an
        example.

    **kwargs: dict of Any
        The keyword arguments that may be passed to mpl.Axes.plot.
    """
    # }}}
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


def plotPCRT(
    figAxTuple: FigAxTuple,
    timeScdsArr: Array,
    pCRTParams: ParameterSequence,
    criticalTime: Optional[float] = None,
    **kwargs: Any,
) -> None:
    # {{{
    """
    Basically a special case of curveFitting.plotFunction that specifically
    plots an exponential function with the given parameters, and a vertical
    line on timeScdsArr[maxDiv] (the critical time 'tc'), and sets the line's
    legend to be about pCRT. See the documentation the aforementioned function
    for more information.
    """
    _, ax = figAxTuple

    funcY = exponential(timeScdsArr, *pCRTParams)

    plotOptions = kwargs.get("plotOptions", None)
    legendOptions = kwargs.get("legendOptions", None)
    if plotOptions is None:
        plotOptions = {"label": "pCRT Exp", "color": "cyan"}
    if legendOptions is None:
        legendOptions = {}
    ax.plot(timeScdsArr, funcY, **plotOptions)

    if criticalTime is not None:
        ax.axvline(criticalTime, label=f"tc={criticalTime:.3g}", c="k", ls=":")

    ax.legend(**legendOptions)


# }}}


def plotCRT9010(
    figAxTuple,
    timeScdsArr,
    avgIntensArr,
    channel,
    crt9010Params,
    criticalTime=None,
):
    # {{{
    _, ax = figAxTuple

    ax.plot(
        timeScdsArr,
        avgIntensArr,
        color=channelColors[channel],
        label=f"Channel {channel.upper()}",
    )

    if criticalTime is not None:
        ax.axvline(criticalTime, label=f"tc={criticalTime:.3g}", c="k", ls=":")

    crt9010, time90, time10, polyCoeffs = crt9010Params

    if polyCoeffs is not None:
        ntimes = len(timeScdsArr)
        polyTimesArr = np.linspace(0, timeScdsArr.max(), 4 * ntimes)
        polyvals = np.polyval(polyCoeffs, polyTimesArr)
        ax.plot(
            polyTimesArr,
            polyvals,
            label="Polynomial interpolation",
            ls="--",
        )

    ax.axvline(time90, label=f"time 90%: {time90:.3g}", c="r", ls=":", lw=2)
    ax.axvline(time10, label=f"time 10%: {time10:.3g}", c="r", ls=":", lw=2)
    ax.legend(loc="upper right")
    addTextToLabel(
        ax,
        f"CRT90-10={crt9010:.2f}",
        loc="upper right",
    )


# }}}


def addTextToLabel(ax: Axes, text: str, **kwargs: Any) -> None:
    # {{{
    """Adds some text to the axes' legend and redraws the legend."""

    handles, _ = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color="none", label=text))
    ax.legend(handles=handles, **kwargs)


# }}}


def makeFigAxes(
    axisLabels: tuple[str, str],
    figTitle: Optional[str] = None,
    figSizePx: tuple[int, int] = (960, 600),
    dpi: Real = 100,
) -> FigAxTuple:
    # {{{
    # {{{
    """
    Creates and formats the figure and axes. To be used with plotFunction,
    plotPCRT and plotAvgIntens.

    Parameters
    ----------
    axisLabels : tuple of str
        The labels for the X and Y axis respectively.

    figTitle : str or None, default=None
        Self explanatory. If None, the figure will just have no title.

    figSizePx : tuple of int, default=(960, 600)
        Figure dimensions in pixels. More adequate for primarily digital
        figures than matplotlib's default of specifying everything in inches.

    dpi : real number, default=100
        The figure's resolution in pixels per inch.

    Returns
    -------
    fig : plt.Figure

    ax : plt.Axes
    """
    # }}}

    xlabel, ylabel = axisLabels

    fig, ax = plt.subplots(
        layout="tight",
        dpi=dpi,
        figsize=tuple(dim / dpi for dim in figSizePx),
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.autoscale(enable=True, axis="x", tight=True)

    ax.xaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    fig.suptitle(figTitle)

    return fig, ax


# }}}


def makeAvgIntensPlot(
    timeScdsArr: Array,
    channelsAvgIntensArr: Array,
    figSizePx=(960, 600),
    channelNames=BGR_INDICES_DICT,
    normIntens=False,
) -> FigAxTuple:
    # {{{

    """
    Creates and formats a plot for the average intensities of all channels over
    all the capture's duration, and returns the Figure and Axes tuple. Not much
    to see here, check out makeFigAxes and plotAvgIntens for more information.
    """

    fig, ax = makeFigAxes(
        ("Time (s)", "Average intensities (a.u.)"),
        "Channel average intensities",
        figSizePx=figSizePx,
    )

    plotAvgIntens(
        (fig, ax),
        timeScdsArr,
        channelsAvgIntensArr,
        channelNames=channelNames,
        normIntens=normIntens,
    )

    return fig, ax


# }}}


def makeCRT9010Plot(
    timeScdsArr,
    avgIntensArr,
    channel,
    crt9010Params,
    criticalTime=None,
    figSizePx=(960, 600),
):
    # {{{
    fig, ax = makeFigAxes(
        ("Time (s)", "Average intensities (a.u.)"),
        "CRT 90-10 calculation",
        figSizePx=figSizePx,
    )

    plotCRT9010(
        (fig, ax),
        timeScdsArr,
        avgIntensArr,
        channel=channel,
        crt9010Params=crt9010Params,
        criticalTime=criticalTime,
    )

    return fig, ax


# }}}


def makePCRTPlot(
    timeScdsArr: Array,
    avgIntensArr: Array,
    funcParamsTuples: dict[str, FitParametersTuple],
    criticalTime: Optional[float] = None,
    channel: Optional[str] = None,
    funcOptions: Optional[dict[str, Any]] = None,
    figSizePx=(960, 600),
) -> FigAxTuple:
    # {{{
    # {{{
    """
    Creates and formats the plot for the exponential, polynomial and pCRT
    exponential functions applied over the array of a channel's intensities
    since the release of the compression on the skin.

    Parameters
    ----------
    timeScdsArr : np.ndarray
        The array of time instants since the release of the compression from
        the skin.

    avgIntenArr : np.ndarray
        The array of average intensities for the channel used for fitting the
        functions.

    funcParamsTuples : dict[str] of tuple
        The dictionary containing the polynomial, exponential and pCRT
        exponential functions' optimized parameters and their respective
        standard deviations. The keys should be 'exponential', 'polynomial' and
        'pCRT' respectively, and the values the tuples returned by
        scipy.optimize.curve_fit. See curveFitting.fitExponential or
        curveFitting.fitPolynomial for more information. If either the
        'polynomial' or 'exponential' keys are lacking, these functions just
        won't be plotted, but the 'pCRT' key is requred.

    criticalTime : float or None, default=None
        The critical time. A vertical dashed line will be drawn to mark this
        instant. If None, no such line will be drawm.

    channel : str or None, default=None
        Which channel to use. This will only be used for the line's color and
        legend, as avgIntenArr is already expected to be of s single channel.
        If None, the legend will be unspecific and the line will be gray.

    funcOptions : dict[str] of any value, default=None
        Additional options that will be passed to the plotting functions
        (plotFunction and plotAvgIntens). The same key naming scheme as with
        the funcParamsTuples parameter is used, but with the addition of the
        optional 'intensities' key.


    Returns
    -------
    fig : matplotlib Figure
        The figure. It is 960x600 pixels by default.

    ax : matplotlib Axes
        The Axes, in which everything is plotted.

    """
    # }}}

    fig, ax = makeFigAxes(
        (
            "Time since release of compression (s)",
            "Average intensities (a.u.)",
        ),
        "Average intensities and fitted functions",
        figSizePx=figSizePx,
    )

    if funcOptions is None:
        funcOptions = {}

    if expTuple := funcParamsTuples.get("exponential", None):
        plotFunction(
            (fig, ax),
            timeScdsArr,
            exponential,
            expTuple[0],
            **funcOptions.get("exponential", {}),
        )
    if polyTuple := funcParamsTuples.get("polynomial", None):
        plotFunction(
            (fig, ax),
            timeScdsArr,
            polynomial,
            polyTuple[0],
            **funcOptions.get("polynomial", {}),
        )

    pCRTTuple = funcParamsTuples["pCRT"]
    plotPCRT(
        (fig, ax),
        timeScdsArr,
        pCRTTuple[0],
        criticalTime,
        **funcOptions.get("pCRT", {}),
    )
    plotAvgIntens(
        (fig, ax),
        timeScdsArr,
        avgIntensArr,
        channel,
        **funcOptions.get("intensities", {}),
    )

    # pCRT, _ = pCRTFromParameters(pCRTTuple)
    pCRT, error = pCRTFromParameters(pCRTTuple)
    relativeUncertainty = calculateRelativeUncertainty(pCRTTuple)

    addTextToLabel(
        ax,
        f"pCRT={pCRT:.2f}±{error:.2f} ({100*relativeUncertainty:.2f}%)",
        loc="upper right",
    )

    return fig, ax


# }}}


def makePlots(
    avgIntensArgs,
    pCRTArgs,
    dpi=100,
    figSizePx=(1200, 600),
    orientation="vertical",
):
    # {{{

    # TODO clean this up
    # omg... thanks, matplotlib...

    if orientation == "vertical":
        fig, (channelsAx, pCRTAx) = plt.subplots(
            ncols=1,
            nrows=2,
            layout="tight",
            dpi=dpi,
            figsize=tuple(dim / dpi for dim in figSizePx),
        )

    elif orientation == "horizontal":
        fig, (channelsAx, pCRTAx) = plt.subplots(
            ncols=2,
            nrows=1,
            layout="tight",
            dpi=dpi,
            figsize=tuple(dim / dpi for dim in figSizePx),
        )

    plotAvgIntens(
        (fig, channelsAx),
        avgIntensArgs["timeScdsArr"],
        avgIntensArgs["channelsAvgIntensArr"],
        channelNames=avgIntensArgs["channelNames"],
        normIntens=avgIntensArgs["normIntens"],
    )
    channelsAx.margins(0, None)
    channelsAx.set_title("Channel average intensities")
    channelsAx.set_xlabel("Time (s)")
    channelsAx.set_ylabel("Average intensities (a.u.)")

    timeScdsArr = pCRTArgs["timeScdsArr"]
    avgIntensArr = pCRTArgs["avgIntensArr"]
    funcParamsTuples = pCRTArgs["funcParamsTuples"]
    criticalTime = pCRTArgs.get("criticalTime", None)
    channel = pCRTArgs.get("channel", None)
    funcOptions = pCRTArgs.get("funcOptions", None)

    if funcOptions is None:
        funcOptions = {}

    if expTuple := funcParamsTuples.get("exponential", None):
        plotFunction(
            (fig, pCRTAx),
            timeScdsArr,
            exponential,
            expTuple[0],
            **funcOptions.get("exponential", {}),
        )
    if polyTuple := funcParamsTuples.get("polynomial", None):
        plotFunction(
            (fig, pCRTAx),
            timeScdsArr,
            polynomial,
            polyTuple[0],
            **funcOptions.get("polynomial", {}),
        )

    pCRTTuple = funcParamsTuples["pCRT"]
    plotPCRT(
        (fig, pCRTAx),
        timeScdsArr,
        pCRTTuple[0],
        criticalTime,
        **funcOptions.get("pCRT", {}),
    )
    plotAvgIntens(
        (fig, pCRTAx),
        timeScdsArr,
        avgIntensArr,
        channel,
        **funcOptions.get("intensities", {}),
    )

    # pCRT, _ = pCRTFromParameters(pCRTTuple)
    pCRT, error = pCRTFromParameters(pCRTTuple)
    relativeUncertainty = calculateRelativeUncertainty(pCRTTuple)

    addTextToLabel(
        pCRTAx,
        f"pCRT={pCRT:.2f}±{error:.2f} ({100*relativeUncertainty:.2f}%)",
        loc="upper right",
    )

    pCRTAx.margins(0, None)
    pCRTAx.set_title("Average intensities and fitted functions")
    pCRTAx.set_xlabel("Time since release of compression (s)")
    pCRTAx.set_ylabel("Average intensities (a.u.)")

    return fig, (channelsAx, pCRTAx)


# }}}


def figVisualizationFunctions(
    func: Callable[..., FigAxTuple],
) -> tuple[Callable[..., None], Callable[..., None]]:
    # {{{
    # {{{
    """
    Creates two functions for easily showing, saving the plot created by the
    'func' function, and closing the plot afterwards.

    Parameters
    ----------
    func: Callable
        The function that actually makes the plot the data. Can be any function
        that takes in any number of arguments and returns a tuple with
        mpl.Figure and mpl.Axes


    Returns
    -------
    showPlot(args, kwargs) : function
        Shows the plot created by func(args, kwargs) and formatted by
        makeFigAxes.

    saveFig(figPath: str, args, kwargs) : function
        Saves the plot created by func(args, kwargs) and formatted by
        makeFigAxes on the specified path.

    Notes
    -----
        This module defines four functions created through
        figVisualizationFactory: showAvgIntensPlot and saveAvgIntensPlot are
        wrappers for makeAvgIntensPlot, and showPCRTPlot and savePCRTPlot are
        wrappers for makePCRTPlot.
        .
    """
    # }}}

    def showPlot(*args, **kwargs) -> None:
        fig, _ = func(*args, **kwargs)
        plt.show()
        if not plt.isinteractive():
            plt.close(fig)

    def saveFig(figPath: str, *args, **kwargs) -> None:
        fig, _ = func(*args, **kwargs)
        plt.savefig(figPath)
        if not plt.isinteractive():
            plt.close(fig)

    return showPlot, saveFig


# }}}


showAvgIntensPlot, saveAvgIntensPlot = figVisualizationFunctions(
    makeAvgIntensPlot
)

showPCRTPlot, savePCRTPlot = figVisualizationFunctions(makePCRTPlot)

showPlots, savePlots = figVisualizationFunctions(makePlots)

showCRT9010Plot, saveCRT9010Plot = figVisualizationFunctions(makeCRT9010Plot)
