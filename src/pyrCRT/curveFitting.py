"""
Functions related to curve fitting (pyrCRT.curveFitting)

This module implements the operations necessary to calculate the rCRT from the average
intensities array and the frame times array, namely fitting a polynomial and two
exponential curves on the data.
"""

from typing import List, Optional, Tuple, Union, overload
from warnings import filterwarnings

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.signal import find_peaks

# This is for catching OptimizeWarnig as if it were an exception
filterwarnings("error")

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
# }}}


def exponential(x: np.ndarray, a: Real, b: Real, c: Real) -> np.ndarray:
    # {{{
    """Exponential function of the form a*exp(b*x)+c. Refer to np.exp from the Numpy
    documentation for more information."""
    return a * np.exp(b * x) + c


# }}}


def polynomial(x: np.ndarray, *coefs: Real) -> np.ndarray:
    # {{{
    """Polynomial of the form coefs[0] + coefs[0]*x + coefs[1]*x**2 + ... Refer to the
    Numpy documentation for more information."""
    return np.polynomial.Polynomial(list(coefs))(x)


# }}}


def covToStdDev(cov: np.ndarray) -> np.ndarray:
    # {{{
    """Converts the covariance matrix returned by SciPy parameter optimization functions
    into an array with the standard deviation of each parameter. Refer to the
    documentation of scipy.optimize.curve_fit for more information"""

    return np.sqrt(np.diag(cov))


# }}}


def fitExponential(
    x: np.ndarray,
    y: np.ndarray,
    p0: Optional[List[Real]] = None,
) -> FitParametersTuple:
    # {{{
    # {{{
    """
    Fits an exponential function of the form a*exp(b*x)+c on the data, and returns a
    tuple of two arrays, one with the optimized parameters and another with their
    standard deviations. Refer to the documentation of scipy.optimize.curve_fit for more
    information.

    Parameters
    ----------
    x, y : np.ndarray
        Self-explanatory.
    p0 : list of 3 real numbers or None, default=None
        The initial guesses for each parameter, in order of a, b and c (see summary
        above). If None, will use p0=[0, 0, 0].

    Returns
    -------
    expParams
        The optimized parameters
    expStdDev
        The optimized parameters' respective standard deviations

    Raises
    ------
    RuntimeError
        If the curve fit failed.
    """
    # }}}

    if p0 is None:
        p0 = [0.0, 0.0, 0.0]

    try:
        expParams, expCov = curve_fit(
            f=exponential,
            xdata=x,
            ydata=y,
            p0=p0,
            bounds=(-np.inf, np.inf),
        )
        expStdDev = covToStdDev(expCov)
        return expParams, expStdDev
    except (RuntimeError, OptimizeWarning) as err:
        raise RuntimeError(f"Exponential fit failed with p0={p0}.") from err


# }}}


def fitPolynomial(
    x: np.ndarray,
    y: np.ndarray,
    p0: Optional[List[Real]] = None,
) -> FitParametersTuple:
    # {{{
    # {{{
    """
    Fits a polynomial function of the form coefs[0] + coefs[0]*x + coefs[1]*x**2 + ...
    on the data, and returns a tuple of two arrays, one with the optimized parameters
    and another with their standard deviations. Refer to the documentation of
    scipy.optimize.curve_fit for more information.

    Parameters
    ----------
    x, y : np.ndarray
        Self-explanatory.
    p0 : list of 3 real numbers or None, default=None
        The initial guesses for each parameter in increasing polynomial order (see
        summary above). Note that this determines the order of the polynomial, for
        example, a list of length 7 specifies a polynomial of sixth order.

    Returns
    -------
    polyParams
        The optimized parameters
    polyStdDev
        The optimized parameters' respective standard deviations

    Raises
    ------
    RuntimeError
        If the curve fit failed.
    """
    # }}}

    if p0 is None:
        p0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    try:
        polyParams, polyCov = curve_fit(
            f=polynomial,
            xdata=x,
            ydata=y,
            p0=p0,
            bounds=(-np.inf, np.inf),
        )
        polyStdDev = covToStdDev(polyCov)
        return polyParams, polyStdDev
    except (RuntimeError, OptimizeWarning) as err:
        raise RuntimeError(f"Polynomial fit failed with p0={p0}.") from err


# }}}


def diffExpPoly(
    x: np.ndarray, expParams: np.ndarray, polyParams: np.ndarray
) -> np.ndarray:
    # {{{
    """
    Evaluates the function |exponential(expParams) - polynomial(polyParams)| over x
    """
    return np.abs(exponential(x, *expParams) - polynomial(x, *polyParams))


# }}}


def fitRCRT(
    x: np.ndarray,
    y: np.ndarray,
    p0: Optional[List[Real]] = None,
    maxDiv: Optional[Union[List[Integer], Integer]] = None,
) -> Tuple[FitParametersTuple, Integer]:
    # {{{
    # {{{
    """
    Slices the x and y arrays from start to maxDiv and fit an exponential function on
    the data, and returns a tuple, the first element of which is the optimized
    parameters and their standard deviations, and the other is the maxDiv used. If
    maxDiv is a list of integers, it will try to fit on each value and return on the
    first successful fit.

    Parameters
    ----------
    x, y : np.ndarray
        Self-explanatory. The arrays over which the curve fit will be tried.
    p0 : list of real numbers, default=None
        The initial guesses for each parameter of the exponential function. Refer to the
        documentation of pyrCRT.curveFitting.exponential for more information.
    maxDiv : list of ints, or int, or None, default=None
        Maximum divergence index between the exponential and polynomial functions fitted
        on the entire data set. Refer to pyrCRT.curveFitting.findMaxDivergencePeaks for
        more information.

    Returns
    -------
    (rCRTParams, rCRTStdDev) : tuple of np.ndarrays
        The optimized parameters for the exponential function and their standard
        deviations.
    maxDiv : int
        The maximum divergence index used. This is useful if a list of maxDivs was
        or None was passed.

    Raises
    ------
    TypeError
        If maxDiv isn't a list, an int, or None.
    RuntimeError
        If the exponential fit failed on the single maxDiv passed, or if a list was
        passed and it failed on all the maxDivs in the list.
    """
    # }}}

    if p0 is None:
        p0 = [0.0, 0.0, 0.0]

    if maxDiv is not None:
        if isinstance(maxDiv, list):
            maxDivList = maxDiv

            for maxDivIndex in maxDivList:
                # Will try fitting with each maxDiv in the list, returning as soon as a
                # fit is successful
                try:
                    return fitRCRT(x, y, maxDiv=maxDivIndex, p0=p0)
                except RuntimeError:
                    pass

            raise RuntimeError(
                f"rCRT fit failed on all maxDivIndexes ({maxDivList})," " with p0={p0}."
            )
        if isinstance(maxDiv, (int, np.int_)):
            maxDivIndex = maxDiv

            try:
                return (
                    fitExponential(x[:int(maxDivIndex)], y[:int(maxDivIndex)], p0=p0),
                    maxDivIndex,
                )
            except RuntimeError as err:
                raise RuntimeError(
                    f"rCRT fit failed on maxDivIndex={maxDivIndex} and p0={p0}"
                ) from err
        raise TypeError(
            f"Invalid type of {type(maxDiv)} for maxDiv. Valid types: int, list of "
            "ints or None. Please refer to the documentation for usage instructions."
        )

    # maxDiv wasn't passed as a kwarg, so this function will try to find the maxDiv
    # itself.
    maxDiv = findMaxDivergencePeaks(x, y)
    return fitRCRT(x, y, maxDiv=maxDiv)


# }}}

# TODO: Move these functions to .arrayOperations as soon as you're done testing this
# module


@overload
def findMaxDivergencePeaks(x: np.ndarray, y: np.ndarray) -> List[Integer]:
    ...


@overload
def findMaxDivergencePeaks(
    x: np.ndarray, expTuple: FitParametersTuple, polyTuple: FitParametersTuple
) -> List[Integer]:
    ...


def findMaxDivergencePeaks(
    x: np.ndarray,
    *args: Union[np.ndarray, FitParametersTuple],
    **kwargs: Union[np.ndarray, FitParametersTuple],
) -> List[Integer]:
    # {{{
    # {{{
    """
    Find the indices of the peaks of maxDiv(expParams, polyParams) and returns them in
    descending order of diffExpPoly[i].
    Usage:
        findMaxDivergencePeaks(x, expParams=expParams, polyParams=polyParams)
    to directly compute the peaks, or
        findMaxDivergencePeaks(x, y)
    to fit the polynomial and exponential functions on the data, and then compute the
    peaks.

    Parameters
    ----------
    x : np.ndarray
        Tipically the array of frame timestamps.
    *args : np.ndarray
        Another array, y, that tipically is the array of average intensities for a
        channel.
    **kwargs : tuple of 2 arrays
        The parameters and standard deviations to the exponential and polynomial
        functions, if they have already been calculated.

    Returns
    -------
    maxIndexesSorted
        List of indexes of x where the peaks of maximum absolute divergence between the
        polynomial and exponential functions have been found, sorted by the peak
        magnitude.
    """
    # }}}

    if "expTuple" in kwargs and "polyTuple" in kwargs:
        expParams, polyParams = kwargs["expTuple"][0], kwargs["polyTuple"][0]
        assert isinstance(expParams, np.ndarray) and isinstance(polyParams, np.ndarray)

        diffArray = diffExpPoly(x, expParams, polyParams)
        maxIndexes = find_peaks(diffArray)[0]
        maxIndexesSorted = sorted(maxIndexes, key=lambda x: diffArray[x], reverse=True)
        return maxIndexesSorted

    if len(args) == 1 and isinstance(args[0], np.ndarray):
        y: np.ndarray = args[0]
        expTuple = fitExponential(x, y)
        polyTuple = fitPolynomial(x, y)
        return findMaxDivergencePeaks(x, expTuple=expTuple, polyTuple=polyTuple)

    raise ValueError(
        "Usage: findMaxDivergencePeaks(x: array, expTuple=expTuple,"
        "polyTuple=polyTuple) or findMaxDivergencePeaks(x: array, y: array)."
        "Please refer to the documentation for more information."
    )


# }}}


def shiftArr(timeArr, arr, **kwargs):
    # {{{
    fromTime = kwargs.get("fromTime", "channel max")
    if fromTime == "channel max":
        fromIndex = np.argmax(arr)
    elif isinstance(fromTime, (int, float)):
        fromIndex = np.where(timeArr >= fromTime)[0][0]

    toTime = kwargs.get("toTime", "end")
    if toTime == "end":
        toIndex = -1
    elif isinstance(toTime, (int, float)):
        toIndex = np.where(timeArr >= toTime)[0][0]

    fromIndex += np.argmax(arr[fromIndex:toIndex])

    return (
        timeArr[fromIndex:toIndex] - np.amin(timeArr[fromIndex:toIndex]),
        arr[fromIndex:toIndex] - np.amin(arr[fromIndex:toIndex]),
    )
