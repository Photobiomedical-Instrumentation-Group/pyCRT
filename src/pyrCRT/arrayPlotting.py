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

# pylint: disable=import-error
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


def plotAvgIntens(
    figAxTuple: FigAxTuple,
    timeScdsArr: np.ndarray,
    avgIntenArr: np.ndarray,
    channels: Optional[str] = None,
    **kwargs: Any,
) -> None:
    # {{{
    # {{{
    """
    Plots the average intensities array on the given mpl.Figure and mpl.Axes tuple. This
    can plot a set of channels or all three, depending on avgIntenArr's dimentions and
    the channels argument.

    Parameters
    ----------
    figAxTuple : tuple of mpl.Figure and mpl.Axes respectively
        The figure and axes on which to plot. In practice this function only utilizes
        the Axes instance, not the Figure.

    timeScdsArr, avgIntenArr : np.ndarray
        The arrays of seconds and average intensities corresponding to each frame,
        respectively. This is typically the output of videoReading.readVideo.

    channels : str or None, default=None
        Which channels to plot, and also used to set the color of each line in the plot
        and its label.Should be a string with some combination of "r", "g" and "b". If
        it is None and avgIntenArr is one-dimensional, then it'll plot that channel in
        gray, and if avgIntenArr is not one-dimensional, then channels will be set as
        "bgr".

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
        if len(avgIntenArr.shape) == 1:  # a single channel
            ax.plot(
                timeScdsArr,
                avgIntenArr,
                color="grey",
                label="Channel",
                **plotOptions,
            )
        else:
            plotAvgIntens(figAxTuple, timeScdsArr, avgIntenArr, channels="bgr")
    else:
        channels = channels.strip().lower()
        if len(channels) == 1:
            ax.plot(
                timeScdsArr,
                avgIntenArr,
                color=channels,
                label=f"Channel {channels.upper()}",
                **plotOptions,
            )
        else:
            for channel in channels.strip().lower():
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
    # {{{
    """
    Applies the given function with the given parameters over timeScdsArr and plots the
    resulting array as the Y axis.

    Parameters
    ----------
    figAxTuple : tuple of mpl.Figure and mpl.Axes respectively
        The figure and axes on which to plot. In practice this function only utilizes
        the Axes instance, not the Figure.

    timeScdsArr : np.ndarray
        The array of seconds corresponding to each frame. This is typically the first
        output of videoReading.readVideo

    func : callable that returns a np.ndarray
        The function that will be applied to timeScdsArr with funcParams as its
        parameters. This is typically the polynomial or exponential functions from
        curveFitting.

    funcParams : list of real numbers
        This list will be unpacked and used as func's positional arguments. Again, see
        curveFitting.exponential or curveFitting.polynomial for an example.

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


def plotRCRT(
    figAxTuple: FigAxTuple,
    timeScdsArr: np.ndarray,
    rCRTParams: List[Real],
    maxDiv: Optional[Integer] = None,
    **kwargs: Any,
) -> None:
    # {{{
    """
    Basically a special case of curveFitting.plotFunction that specifically plots an
    exponential function with the given parameters, and a vertical line on
    timeScdsArr[maxDiv], and sets the line's legend to be about rCRT. See the
    documentation the aforementioned function for more information.
    """
    _, ax = figAxTuple

    funcY = exponential(timeScdsArr, *rCRTParams)

    plotOptions = kwargs.get("plotOptions", None)
    legendOptions = kwargs.get("legendOptions", None)
    if plotOptions is None:
        plotOptions = {"label": "rCRT Exp", "color": "cyan"}
    if legendOptions is None:
        legendOptions = {}
    ax.plot(timeScdsArr, funcY, **plotOptions)

    if maxDiv is not None:
        ax.axvline(timeScdsArr[maxDiv], label="maxDiv", c="k", ls=":")

    ax.legend(**legendOptions)


# }}}


# This is a filthy hack
def _plotAvgIntensAndFunctions(
    figAxTuple: FigAxTuple,
    timeScdsArr: np.ndarray,
    avgIntenArr: np.ndarray,
    funcParams: Dict[str, List[Real]],
    maxDiv: Optional[Integer] = None,
    funcOptions: Optional[Dict[str, Any]] = None,
) -> None:
    # {{{
    # {{{
    """
    Just calls plotAvgIntens, plotFunction twice and plotRCRT with the parameters given
    by the funcParams dictionary and keyword arguments given by the funcOptions
    dictionary of dictionaries as their respective keyword arguments.

    Parameters
    ----------
    figAxTuple : tuple of mpl.Figure and mpl.Axes respectively
        The figure and axes on which to plot. In practice this function only utilizes
        the Axes instance, not the Figure.

    timeScdsArr, avgIntenArr : np.ndarray
        The arrays of seconds and average intensities corresponding to each frame,
        respectively. This is typically the output of videoReading.readVideo.

    funcParams : dict of lists of real numbers, or None
        A dictionary with the funcParams for each function (see plotFunction for more
        information). The keys should be 'exponential', 'polynomial', 'rCRT' and
        'intensities'.

    maxDiv : integer or None
        the maximum divergence index, a vertical line will be plotted on
        timeScdsArr[maxDiv] if it is not None. See curveFitting.findMaxDivergencePeaks
        and curveFitting.fitRCRT for more information.

    funcOptions: dict of dicts of Any, or None
        A dictionary of dictionaries, each key corresponding to a dictionary of keyword
        arguments to be passed to the each function. It accepts the same keys as
        funcParams.

    Notes
    -----
        This is a very convoluted and difficult to use function. There's no reason to
        directly call this function over calling each function it calls individually,
        except for hacky workarounds such as what I've done with
        figVisualizationFactory, which is the entire reason this function exists and is
        documented in the first place.

    """
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


def makeFigAxes(
    axisLabels: Tuple[str, str],
    figTitle: Optional[str] = None,
    figSizePx: Tuple[int, int] = (960, 600),
    dpi: Real = 100,
) -> FigAxTuple:
    # {{{
    # {{{
    """
    Creates and formats the figure and axes. To be used with plotFunction, plotRCRT and
    plotAvgIntens.

    Parameters
    ----------
    axisLabels : tuple of str
        The labels for the X and Y axis respectively.

    figTitle : str or None, default=None
        Self explanatory. If None, the figure will just have no title.

    figSizePx : tuple of int, default=(960, 600)
        Figure dimensions in pixels. More adequate for primarily digital figures than
        matplotlib's default of specifying everything in inches.

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
        constrained_layout=True,
        dpi=dpi,
        figsize=tuple(dim / dpi for dim in figSizePx),
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.autoscale(enable=True, axis="x", tight=True)

    fig.suptitle(figTitle)

    return fig, ax


# }}}


def figVisualizationFactory(
    func: Callable,
    figTitle: Optional[str] = None,
    axisLabels: Tuple[str, str] = ("Time (s)", "Average Intensities (a.u.)"),
    figSizePx: Tuple[int, int] = (960, 600),
) -> Tuple[Callable, Callable]:
    # {{{
    # {{{
    """
    Creates two functions for easily showing, saving the plot created by the 'func'
    function, and closing the plot afterwards.

    Parameters
    ----------
    func: Callable
        The function that actually plots the data. Can be plotRCRT, plotFunction,
        plotAvgIntens, or _plotAvgIntensAndFunctions, or any function of that sort that
        the user may implement.

    axisLabels: tuple of str, default=('Time (s)', 'Average Intensities (a.u.)')
        The labels for the X and Y axis respectively.

    figTitle : str or None, default=None
        Self explanatory. If None, the figure will just have no title.

    figSizePx : tuple of int, default=(960, 600)
        Figure dimensions in pixels. More adequate for primarily digital figures than
        matplotlib's default of specifying everything in inches.

    Returns
    -------
    showPlot(*args, **kwargs) : function
        Shows the plot created by func(*args, **kwargs) and formatted by makeFigAxes.

    saveFig(figPath: str, *args, **kwargs) : function
        Saves the plot created by func(*args, **kwargs) and formatted by makeFigAxes on
        the specified path.

    Notes
    -----
        This module defines four functions created through figVisualizationFactory:
        showAvgInten and saveAvgInten are wrappers for plotAvgIntens, and
        showAvgIntenAndFunctions and saveAvgIntensAndFunctions are wrappers for
        _plotAvgIntensAndFunctions.
    """
    # }}}

    def showPlot(*args, **kwargs) -> None:
        try:
            fig, ax = makeFigAxes(axisLabels, figTitle, figSizePx)
            func((fig, ax), *args, **kwargs)
        finally:
            if not plt.isinteractive():
                plt.show()
                plt.close(fig)

    def saveFig(figPath: str, *args, **kwargs) -> None:
        try:
            fig, ax = makeFigAxes(axisLabels, figTitle, figSizePx)
            func((fig, ax), *args, **kwargs)
            plt.savefig(figPath)
        finally:
            if not plt.isinteractive():
                plt.close(fig)

    return showPlot, saveFig


# }}}


showAvgIntens, saveAvgIntens = figVisualizationFactory(
    plotAvgIntens, "Channel average intensities"
)

showAvgIntensAndFunctions, saveAvgIntensAndFunctions = figVisualizationFactory(
    _plotAvgIntensAndFunctions, "Average Intensities and fitted functions"
)
