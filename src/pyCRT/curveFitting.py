"""
Functions related to curve fitting (pyCRT.curveFitting)

This module implements the operations necessary to calculate the pCRT from the
average intensities array and the frame times array, namely fitting a
polynomial and two exponential curves on the data.
"""

from typing import Iterable, Optional, Sequence, Tuple, Union, overload
from warnings import filterwarnings

import numpy as np

from numpy.typing import NDArray
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.signal import find_peaks

from .arrayOperations import findValueIndex

# This is for catching OptimizeWarnig as if it were an exception
filterwarnings("error")

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
# }}}


def exponential(x: Array, a: Real, b: Real, c: Real) -> Array:
    # {{{
    """Exponential function of the form a*exp(b*x)+c. Refer to np.exp from the
    Numpy documentation for more information."""
    return a * np.exp(b * x) + c


# }}}


def polynomial(x: Array, *coefs: Real) -> Array:
    # {{{
    """Polynomial of the form coefs[0] + coefs[0]*x + coefs[1]*x**2 + ... Refer
    to the Numpy documentation for more information."""
    return np.polynomial.Polynomial(list(coefs))(x)


# }}}


def covToStdDev(cov: Array) -> Array:
    # {{{
    """Converts the covariance matrix returned by SciPy parameter optimization
    functions into an array with the standard deviation of each parameter.
    Refer to the documentation of scipy.optimize.curve_fit for more
    information"""

    return np.sqrt(np.diag(cov))


# }}}


def fitExponential(
    x: Array,
    y: Array,
    p0: Optional[ParameterSequence] = None,
) -> tuple[Array, Array]:
    # {{{
    # {{{
    """
    Fits an exponential function of the form a*exp(b*x)+c on the data, and
    returns a tuple of two arrays, one with the optimized parameters and
    another with their standard deviations. Refer to the documentation of
    scipy.optimize.curve_fit for more information.

    Parameters
    ----------
    x, y : np.ndarray
        Self-explanatory.
    p0 : sequence of 3 real numbers or None, default=None
        The initial guesses for each parameter, in order of a, b and c (see
        summary above). If None, will use p0=[1.0, -0.3, 0].

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
        p0 = [1.0, -0.3, 0.0]

    try:
        # pylint: disable=unbalanced-tuple-unpacking
        expParams, expCov = curve_fit(
            f=exponential,
            xdata=x,
            ydata=y,
            p0=p0,
            bounds=([0.0, -np.inf, -np.inf], [np.inf, 0.0, np.inf]),
            full_output=False,
        )
        expStdDev = covToStdDev(expCov)
        return expParams, expStdDev
    except (RuntimeError, OptimizeWarning) as err:
        raise RuntimeError(
            f"Exponential fit failed with p0={np.array(p0)}."
        ) from err


# }}}


def fitPolynomial(
    x: Array,
    y: Array,
    p0: Optional[ParameterSequence] = None,
) -> tuple[Array, Array]:
    # {{{
    # {{{
    """
    Fits a polynomial function of the form coefs[0] + coefs[0]*x +
    coefs[1]*x**2 + ... on the data, and returns a tuple of two arrays, one
    with the optimized parameters and another with their standard deviations.
    Refer to the documentation of scipy.optimize.curve_fit for more
    information.

    Parameters
    ----------
    x, y : np.ndarray
        Self-explanatory.
    p0 : sequence of real numbers or None, default=None
        The initial guesses for each parameter in increasing polynomial order
        (see summary above). Note that this determines the order of the
        polynomial, for example, a list of length 7 specifies a polynomial of
        sixth order.

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
        # pylint: disable=unbalanced-tuple-unpacking
        (
            polyParams,
            polyCov,
        ) = curve_fit(
            f=polynomial,
            xdata=x,
            ydata=y,
            p0=p0,
            bounds=(-np.inf, np.inf),
            full_output=False,
        )
        polyStdDev = covToStdDev(polyCov)
        return polyParams, polyStdDev
    except (RuntimeError, OptimizeWarning) as err:
        raise RuntimeError(
            f"Polynomial fit failed with p0={np.array(p0)}."
        ) from err


# }}}


def diffExpPoly(x: Array, expParams: Array, polyParams: Array) -> Array:
    # {{{
    """
    Evaluates the function abs(exponential(expParams) - polynomial(polyParams))
    over x
    """
    return np.abs(exponential(x, *expParams) - polynomial(x, *polyParams))


# }}}


def fitPCRT(
    x: Array,
    y: Array,
    p0: Optional[ParameterSequence] = None,
    maxDiv: Optional[Union[Iterable[int], int]] = None,
) -> tuple[tuple[Array, Array], int]:
    # {{{
    # {{{
    """
    Slices the x and y arrays from start to maxDiv and fit an exponential
    function on the data, and returns a tuple, the first element of which is
    the optimized parameters and their standard deviations, and the other is
    the maxDiv used. If maxDiv is a list of integers, it will try to fit on
    each value and return on the first successful fit.

    Parameters
    ----------
    x, y : np.ndarray
        Self-explanatory. The arrays over which the curve fit will be tried.
    p0 : sequence of real numbers, default=None
        The initial guesses for each parameter of the exponential function.
        Refer to the documentation of pyCRT.curveFitting.exponential for more
        information.
    maxDiv : iterable of ints, or int, or None, default=None
        Maximum divergence index between the exponential and polynomial
        functions fitted on the entire data set. Refer to
        pyCRT.curveFitting.findMaxDivergencePeaks for more information.

    Returns
    -------
    (pCRTParams, pCRTStdDev) : tuple of np.ndarrays
        The optimized parameters for the exponential function and their
        standard deviations.
    maxDiv : int
        The maximum divergence index used. This is useful if a list of maxDivs
        was or None was passed.

    Raises
    ------
    TypeError
        If maxDiv isn't a list, an int, or None.
    RuntimeError
        If the exponential fit failed on the single maxDiv passed, or if a list
        was passed and it failed on all the maxDivs in the list.
    """
    # }}}

    if p0 is None:
        p0 = [1.0, -0.3, 0.0]

    if maxDiv is not None:
        if isinstance(maxDiv, list):
            maxDivList = maxDiv

            for maxDivIndex in maxDivList:
                # Will try fitting with each maxDiv in the list, returning as
                # soon as a fit is successful
                try:
                    return fitPCRT(x, y, maxDiv=maxDivIndex, p0=p0)
                except RuntimeError:
                    pass

            raise RuntimeError(
                f"pCRT fit failed on all maxDivIndexes ({maxDivList}),"
                " with p0={p0}."
            )
        if isinstance(maxDiv, int):
            maxDivIndex = maxDiv

            try:
                return (
                    fitExponential(
                        x[: int(maxDivIndex)], y[: int(maxDivIndex)], p0=p0
                    ),
                    maxDivIndex,
                )
            except RuntimeError as err:
                raise RuntimeError(
                    f"pCRT fit failed on maxDivIndex={maxDivIndex} "
                    f"and p0={np.array(p0)}"
                ) from err
        raise TypeError(
            f"Invalid type of {type(maxDiv)} for maxDiv. Valid types: int, "
            "list of ints or None. Please refer to the documentation for "
            "usage instructions."
        )

    # maxDiv wasn't passed as a kwarg, so this function will try to find the
    # maxDiv itself.
    maxDiv = findMaxDivergencePeaks(x, y)
    return fitPCRT(x, y, maxDiv=maxDiv)


# }}}


def pCRTFromParameters(pCRTTuple: FitParametersTuple) -> tuple[float, float]:
    # {{{
    # {{{
    """
    Calculate the pCRT and its uncertainty with a 95% confidence interval from
    the pCRT exponential's optimized parameters.

    Parameters
    ----------
    pCRTTuple : tuple of sequences of float
        A tuple with the fitted parameters and their standard deviations,
        respectively. See fitPCRT.

    Returns
    -------
    pCRT : float
        The calculated pCRT, which is the negative inverse of the "b" parameter
        of the exponential function defined in this module (see exponential).

    pCRTUncertainty : float
        The pCRT's uncertainty with a 95% confidence interval, calculated from
        the standard deviation of the "b" parameter of the exponential
        function.
    """
    # }}}

    pCRTParams, pCRTStdDev = pCRTTuple

    inversePCRT: float = pCRTParams[1]
    inversePCRTStdDev: float = pCRTStdDev[1]

    pCRT = -1 / inversePCRT
    pCRTUncertainty = -2 * pCRT * (inversePCRTStdDev / inversePCRT)

    return (pCRT, pCRTUncertainty)


# }}}


def calculateRelativeUncertainty(pCRTTuple: FitParametersTuple) -> np.float_:
    # {{{
    """
    Calculates the pCRT's relative uncertainty (with a 95% confidence interval)
    given a tuple with the optimized pCRT exponential parameters and their
    respective standard deviations.
    """

    pCRTParams, pCRTStdDev = pCRTTuple
    return 2 * abs(pCRTStdDev[1] / pCRTParams[1])


# }}}


@overload
def findMaxDivergencePeaks(x: Array, y: Array) -> list[int]:
    ...


@overload
def findMaxDivergencePeaks(
    x: Array, expTuple: FitParametersTuple, polyTuple: FitParametersTuple
) -> list[int]:
    ...


def findMaxDivergencePeaks(
    x: Array,
    *args: Union[Array, FitParametersTuple],
    **kwargs: Union[Array, FitParametersTuple],
) -> list[int]:
    # {{{
    # {{{
    """
    Find the indices of the peaks of maxDiv(expParams, polyParams) and returns
    them in descending order of diffExpPoly[i].
    Usage:

        findMaxDivergencePeaks(x, expParams=expParams, polyParams=polyParams)


    to directly compute the peaks, or

        findMaxDivergencePeaks(x, y)


    to fit the polynomial and exponential functions on the data, and then
    compute the peaks.

    Parameters
    ----------
    x : np.ndarray
        Tipically the array of frame timestamps.
    *args : np.ndarray
        Another array, y, that tipically is the array of average intensities
        for a channel.
    **kwargs : tuple of 2 arrays
        The parameters and standard deviations to the exponential and
        polynomial functions, if they have already been calculated.

    Returns
    -------
    maxIndexesSorted
        List of indexes of x where the peaks of maximum absolute divergence
        between the polynomial and exponential functions have been found,
        sorted by the peak magnitude.
    """
    # }}}

    if "expTuple" in kwargs and "polyTuple" in kwargs:
        expParams, polyParams = kwargs["expTuple"][0], kwargs["polyTuple"][0]
        assert isinstance(
            expParams, np.ndarray
        ) and expParams.dtype == np.dtype("float64")
        assert isinstance(
            polyParams, np.ndarray
        ) and polyParams.dtype == np.dtype("float64")

        diffArray = diffExpPoly(x, expParams, polyParams)
        maxIndexes = find_peaks(diffArray)[0]
        maxIndexes = [int(x) for x in maxIndexes]
        maxIndexesSorted = sorted(
            maxIndexes, key=lambda x: diffArray[x], reverse=True
        )
        return maxIndexesSorted

    if len(args) == 1 and isinstance(args[0], np.ndarray):
        y: Array = args[0]
        expTuple = fitExponential(x, y)
        polyTuple = fitPolynomial(x, y)
        return findMaxDivergencePeaks(
            x, expTuple=expTuple, polyTuple=polyTuple
        )

    raise ValueError(
        "Usage: findMaxDivergencePeaks(x: array, expTuple=expTuple,"
        "polyTuple=polyTuple) or findMaxDivergencePeaks(x: array, y: array)."
        "Please refer to the documentation for more information."
    )


# }}}


def calcPCRT(
    timeScdsArr: Array,
    avgIntensArr: Array,
    criticalTime: Optional[Union[float, Iterable[float]]] = None,
    expTuple: Optional[FitParametersTuple] = None,
    polyTuple: Optional[FitParametersTuple] = None,
    pCRTInitialGuesses: Optional[ParameterSequence] = None,
    exclusionMethod: str = "best fit",
    exclusionCriteria: float = np.inf,
) -> Tuple[ArrayTuple, float]:
    # {{{
    # {{{
    """
    Fits the pCRT exponential on the f(timeScds)=avgIntens data given by
    timeScdsArr and avgIntensArr respectively on each value of criticalTime,
    returning the fitted pCRT parameters and their standard deviations chosen
    according to the exlucion method and criteria (see Parameters below).


    Parameters
    ----------
    timeScdsArr : np.ndarray of float
        An array of time instants in seconds. Typically corresponding to the
        timestamp of each frame in a video recording.

    avgIntensArr : np.ndarray of float
        The array of average intensities for a given channel inside the region
        of interest (ROI), with respect to timeScdsArr.

    criticalTime : float, iterable of float, or None, default=None
        The critical time, up until which the pCRT function will be optimized
        on the timeScdsArr and avgIntensArr. If a list of float, this function
        will try fitting the pCRT exponential function on each criticalTime and
        return the optimized parameters chosen according to exclusionMethod and
        exclusionCriteria. If None, curveFitting.findMaxDivergencePeaks will be
        called on timeScdsArr and avgIntensArr to find a list of candidate
        criticalTimes.

    expTuple, polyTuple : tuple of np.ndarray of float, default=None
        Tuples with the exponential and polynomial function parameters and
        standard deviations fitted over f(timeScds)=avgIntens. These arguments
        will be used to find the list of candidate critical times if expTuple
        and polyTuple are not None, and criticalTime is None.

    pCRTInitialGuesses : sequence of float or None, default=None
        The initial guesses for the rRCT exponential fitting. If None, p0=[1.0,
        -0.3, 0.0] will be used by default (see curveFitting.fitPCRT and
        curveFitting.fitExponential).

    exclusionMethod : str, default='best fit'
        Which criticalTime and its associated fitted pCRT parameters and
        standard deviations are to be returned. Possible values are 'best fit',
        'strict' and 'first that works' (consult the documentation for the
        calcPCRTBestFit, calcPCRTStrict and calcPCRTFirstThatWorks functions
        for a description of the effect of these possible values). Of course,
        this parameter has no effect if a single criticalTime is provided
        (instead of a list of candidate criticalTimes or none at all)

    exclusionCriteria : float, default=np.inf
        The maximum relative uncertainty a pCRT measurement can have and not be
        rejected. If all fits on the criticalTime candidates fail this
        criteria, a RuntimeError will be raised.

    Returns
    -------
    pCRTTuple : tuple of np.ndarray of float
        The optimized parameters and their standard deviations, respectively,
        chosen according to the exclusionMethod and exclusionCriteria.

    criticalTime : float
        The critical time, chosen according to the exclusionMethod and
        exclusionCriteria.

    Raises
    ------
    ValueError
        If an invalid value for exclusionMethod was passed.

    RuntimeError
        If either the fit failed on all criticalTime candidates or no fit
        passed the exclusionCriteria.

    See Also
    --------
    calcPCRTBestFit, calcPCRTStrict, calcPCRTFirstThatWorks :
        calcPCRT only serves to compute a list of critical time candidates (if
        it isn't provided) and call these these functions to deal with each
        possible value of exclusionMethod.

    """
    # }}}

    if criticalTime is None:
        if expTuple is None or polyTuple is None:
            maxDivList = findMaxDivergencePeaks(timeScdsArr, avgIntensArr)
            criticalTimeList = list(timeScdsArr[maxDivList])

        else:
            maxDivList = findMaxDivergencePeaks(
                timeScdsArr, expTuple=expTuple, polyTuple=polyTuple
            )
            criticalTimeList = list(timeScdsArr[maxDivList])

    else:
        if isinstance(criticalTime, float):
            return calcPCRTStrict(
                timeScdsArr,
                avgIntensArr,
                criticalTime,
                pCRTInitialGuesses,
                exclusionCriteria,
            )

        if isinstance(criticalTime, Iterable):
            criticalTimeList = list(criticalTime)

        else:
            raise TypeError(
                f"Invalid type ({criticalTime}) of criticalTime passed. "
                "Valid types: float, list of float or None."
            )

    if exclusionMethod == "best fit":
        return calcPCRTBestFit(
            timeScdsArr,
            avgIntensArr,
            criticalTimeList,
            pCRTInitialGuesses,
            exclusionCriteria,
        )

    if exclusionMethod == "strict":
        return calcPCRTStrict(
            timeScdsArr,
            avgIntensArr,
            criticalTimeList[0],
            pCRTInitialGuesses,
            exclusionCriteria,
        )

    if exclusionMethod == "first that works":
        return calcPCRTFirstThatWorks(
            timeScdsArr,
            avgIntensArr,
            criticalTimeList,
            pCRTInitialGuesses,
            exclusionCriteria,
        )

    raise ValueError(
        f"Invalid value of {exclusionMethod} passed as exclusionMethod. "
        "Valid values: 'best fit', 'strict' and 'first that works'."
    )


# }}}


def calcPCRTBestFit(
    timeScdsArr: Array,
    avgIntensArr: Array,
    criticalTimeList: Iterable[float],
    pCRTInitialGuesses: Optional[ParameterSequence] = None,
    exclusionCriteria: float = np.inf,
) -> Tuple[ArrayTuple, float]:
    # {{{
    # {{{
    """
    Fits the pCRT exponential on the f(timeScds)=avgIntens data given by
    timeScdsArr and avgIntensArr respectively using the candidate critical
    times given by criticalTimeList, returning the fitted parameters, their
    standard deviations and the criticalTime that gave the lowest relative
    uncertainty for the 1/pCRT parameter.

    Parameters
    ----------
    timeScdsArr : np.ndarray of float
        An array of time instants in seconds. Typically corresponding to the
        timestamp of each frame in a video recording.

    avgIntensArr : np.ndarray of float
        The array of average intensities for a given channel inside the region
        of interest (ROI), with respect to timeScdsArr.

    criticalTimeList : iterable of float
        An iterable of candidate critical times. curveFitting.fitPCRT will be
        called with each criticalTime.

    pCRTInitialGuesses : sequence of float or None, default=None
        The initial guesses for the rRCT exponential fitting. If None, p0=[1.0,
        -0.3, 0.0] will be used by default (see curveFitting.fitPCRT and
        curveFitting.fitExponential).

    exclusionCriteria : float, default=np.inf
        The maximum relative uncertainty a pCRT measurement can have and not be
        rejected. If all fits on the criticalTime candidates fail this
        criteria, a RuntimeError will be raised.

    Returns
    -------
    pCRTTuple : tuple of np.ndarray of float
        The optimized parameters and their standard deviations, respectively,
        that gave the least standard deviation for the 1/pCRT parameter of the
        pCRT exponential function.

    criticalTime : float
        The critical time associated with the aforementioned parameters.

    Raises
    ------
    RuntimeError
        If either the fit failed on all criticalTime candidates or no fit
        passed the exclusionCriteria.

    """
    # }}}

    # A dictionary whose keys are maximum divergence indexes (maxDivs) and
    # values the pCRT and its uncertainty calculated with the respective maxDiv
    maxDivResults: dict[int, ArrayTuple] = {}
    for criticalTime in criticalTimeList:
        maxDiv = findValueIndex(timeScdsArr, criticalTime)
        try:
            pCRTTuple, maxDiv = fitPCRT(
                timeScdsArr,
                avgIntensArr,
                pCRTInitialGuesses,
                maxDiv,
            )
            maxDivResults[maxDiv] = pCRTTuple
        except RuntimeError:
            pass

    if not maxDivResults:
        raise RuntimeError(
            "pCRT fit failed on all critical times: {criticalTimeList} "
            f"with initial guesses = {pCRTInitialGuesses}"
        )

    maxDiv = min(
        maxDivResults,
        key=lambda x: calculateRelativeUncertainty(maxDivResults[x]),
    )
    pCRTTuple = maxDivResults[maxDiv]

    relativeUncertainty = calculateRelativeUncertainty(maxDivResults[maxDiv])

    if relativeUncertainty > exclusionCriteria:
        raise RuntimeError(
            "Resulting pCRT parameters did not pass the exclusion criteria of "
            f"{exclusionCriteria}. Relative uncertainty:"
            f" {relativeUncertainty}."
        )

    return pCRTTuple, timeScdsArr[maxDiv]


# }}}


def calcPCRTStrict(
    timeScdsArr: Array,
    avgIntensArr: Array,
    criticalTime: float,
    pCRTInitialGuesses: Optional[ParameterSequence] = None,
    exclusionCriteria: float = np.inf,
) -> Tuple[ArrayTuple, float]:
    # {{{
    # {{{
    """
    Fits the pCRT exponential on the f(timeScds)=avgIntens data given by
    timeScdsArr and avgIntensArr respectively using criticalTime as the
    critical time, returning the fitted parameters, their standard deviations
    and the critical time itself.

    Parameters
    ----------
    timeScdsArr : np.ndarray of float
        An array of time instants in seconds. Typically corresponding to the
        timestamp of each frame in a video recording.

    avgIntensArr : np.ndarray of float
        The array of average intensities for a given channel inside the region
        of interest (ROI), with respect to timeScdsArr.

    criticalTime : float
        The critical time. The pCRT exponential will be fitted on timeScdsArr
        and avgIntenArr up until this value in timeScdsArr.

    pCRTInitialGuesses : sequence of float or None, default=None
        The initial guesses for the rRCT exponential fitting. If None, p0=[1.0,
        -0.3, 0.0] will be used by default (see curveFitting.fitPCRT and
        curveFitting.fitExponential).

    exclusionCriteria : float, default=np.inf
        The maximum relative uncertainty a pCRT measurement can have and not be
        rejected. If all fits on the criticalTime candidates fail this
        criteria, a RuntimeError will be raised.

    Returns
    -------
    pCRTTuple : tuple of np.ndarray of float
        The optimized parameters and their standard deviations

    criticalTime : float
        The critical time associated with the aforementioned parameters.

    Raises
    ------
    RuntimeError
        If either the fit failed or it didn't pass the exclusion criteria.
    """
    # }}}

    pCRTTuple, maxDiv = fitPCRT(
        timeScdsArr,
        avgIntensArr,
        pCRTInitialGuesses,
        findValueIndex(timeScdsArr, criticalTime),
    )

    relativeUncertainty = calculateRelativeUncertainty(pCRTTuple)

    if relativeUncertainty > exclusionCriteria:
        raise RuntimeError(
            "Resulting pCRT parameters did not pass the exclusion criteria of "
            f"{exclusionCriteria}. Relative uncertainty: "
            f"{relativeUncertainty}."
        )

    return pCRTTuple, timeScdsArr[maxDiv]


# }}}


def calcPCRTFirstThatWorks(
    timeScdsArr: Array,
    avgIntensArr: Array,
    criticalTimeList: Iterable[float],
    pCRTInitialGuesses: Optional[ParameterSequence] = None,
    exclusionCriteria: float = np.inf,
) -> Tuple[ArrayTuple, float]:
    # {{{
    # {{{
    """
    Fits the pCRT exponential on the f(timeScds)=avgIntens data given by
    timeScdsArr and avgIntensArr respectively using the candidate critical
    times given by criticalTimeList, returning the fitted parameters, their
    standard deviations and the criticalTime corresponding to the first
    critical time candidate that passed the exclusion criteria.

    Parameters
    ----------
    timeScdsArr : np.ndarray of float
        An array of time instants in seconds. Typically corresponding to the
        timestamp of each frame in a video recording.

    avgIntensArr : np.ndarray of float
        The array of average intensities for a given channel inside the region
        of interest (ROI), with respect to timeScdsArr.

    criticalTimeList : iterable of float
        An iterable of candidate critical times. curveFitting.fitPCRT will be
        called with each criticalTime.

    pCRTInitialGuesses : sequence of float or None, default=None
        The initial guesses for the rRCT exponential fitting. If None, p0=[1.0,
        -0.3, 0.0] will be used by default (see curveFitting.fitPCRT and
        curveFitting.fitExponential).

    exclusionCriteria : float, default=np.inf
        The maximum relative uncertainty a pCRT measurement can have and not be
        rejected. If all fits on the criticalTime candidates fail this
        criteria, a RuntimeError will be raised.

    Returns
    -------
    pCRTTuple : tuple of np.ndarray of float
        The first optimized parameters and their respective standard deviations
        that passed the exclusion criteria.

    criticalTime : float
        The critical time associated with the aforementioned parameters.

    Raises
    ------
    RuntimeError
        If either the fit failed on all criticalTime candidates or no fit
        passed the exclusionCriteria.

    """
    # }}}

    criticalTimeList = sorted(criticalTimeList)
    for criticalTime in criticalTimeList:
        maxDiv = findValueIndex(timeScdsArr, criticalTime)
        try:
            pCRTTuple, maxDiv = fitPCRT(
                timeScdsArr,
                avgIntensArr,
                pCRTInitialGuesses,
                maxDiv,
            )

            relativeUncertainty = calculateRelativeUncertainty(pCRTTuple)
            if relativeUncertainty < exclusionCriteria:
                return pCRTTuple, timeScdsArr[maxDiv]

        except RuntimeError:
            pass

    raise RuntimeError(
        f"No pCRT parameters passed the exclusion criteria of "
        f"{exclusionCriteria}, with initial guesses = {pCRTInitialGuesses}"
        f"and critical time candidates = "
        f"{criticalTimeList}."
    )


# }}}
