# Quick script to fit all the functions on the test data.
# Only meant to be used for testing the other modules on ipython.

# pylint: disable=import-error
from _readTestData import G, timeScdsArr

# pylint: disable=import-error
from curveFitting import fitExponential, fitPolynomial, fitRCRT, findMaxDivergencePeaks

# pylint: disable=import-error
from arrayOperations import sliceFromMaxToEnd, minMaxNormalize

timeScdsArr, G = sliceFromMaxToEnd(timeScdsArr, G)
G = minMaxNormalize(G)

expTuple = fitExponential(timeScdsArr, G)
polyTuple = fitPolynomial(timeScdsArr, G)

maxDiv = findMaxDivergencePeaks(timeScdsArr, expTuple=expTuple, polyTuple=polyTuple)

rCRTTuple, maxDiv = fitRCRT(timeScdsArr, G, maxDiv)

expParams, expStdDev = expTuple
polyParams, polyStdDev = polyTuple
rCRTParams, rCRTStdDev = rCRTTuple

funcParams = {"exponential": expParams, "polynomial": polyParams, "rCRT": rCRTParams}
