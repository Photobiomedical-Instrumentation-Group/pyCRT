"""
A simplified object-oriented interface for pyrCRT (pyrCRT.simpleUI)

This module provides the RCRT class, which is meant to be the simplest way to use
pyrCRT's functions distributed among it's other modules.
"""

from arrayOperations import minMaxNormalize, sliceFromLocalMax
from curveFitting import findMaxDivergencePeaks, fitRCRT, rCRTFromParameters

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
# This accounts for the fact that np.int_ doesn't inherit from int
Integer = Union[int, np.int_]

# Tuple of two lists, the first being the fitted parameters and the second their
# standard deviations
FitParametersTuple = Tuple[np.ndarray, np.ndarray]

# Function that slices the timeScdsArr and avgIntenArr
SlicingFunction = Callable[..., ArrayTuple]
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
        **kwargs: Any,
    ):
        self.fullTimeScds = fullTimeScds
        self.fullAvgIntens = fullAvgIntens
        splitChannels(fullTimeScds)

        self.usedChannel = channelToUse.strip().lower()
        self.channelAvgIntens = self.fullAvgIntens[
            :, CHANNEL_INDICES_DICT[self.usedChannel]
        ]

        self.timeScds, self.avgIntens = sliceFromLocalMax(
            self.fullTimeScds, self.channelAvgIntens, fromTime, toTime
        )

    def calcRCRT(
        self, initialGuesses: Dict[str, List[Real]] = {}
    ) -> FitParametersTuple:

        self.expParams, self.expStdDev = fitExponential(
            self.timeScds, self.avgIntens, initialGuesses.get("exponential", None)
        )

        self.polyParams, self.polyStdDev = fitPolynomial(
            self.timeScds, self.avgIntens, initialGuesses.get("polynomial", None)
        )

        maxDivergencePeaks = findMaxDivergencePeaks(
            self.timeScds, self.expParams, self.polyParams
        )

        for maxDiv in maxDivergencePeaks:
            try:
                (rCRTParams, rCRTStdDev), maxDiv = fitRCRT(
                    self.timeScds,
                    self.avgIntens,
                    initialGuesses.get("rCRT", None),
                    maxDiv,
                )
            except RuntimeError:
                pass

            rCRT, rCRTUncertainty = rCRTFromParameters((rCRTParams, rCRTStdDev))
