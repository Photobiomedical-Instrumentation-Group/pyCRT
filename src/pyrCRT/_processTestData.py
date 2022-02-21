# Quick script to fit all the functions on the test data.
# Only meant to be used for testing the other modules on ipython.

# pylint: disable=import-error
from _readTestData import G, timeScdsArr

# pylint: disable=import-error
from curveFitting import fitExponential, fitPolynomial, fitRCRT, findMaxDivergencePeaks

# pylint: disable=import-error
from arrayOperations import shiftArr

timeShifted, GShifted = shiftArr(timeScdsArr, G)

expTuple = fitExponential(timeShifted, GShifted)
polyTuple = fitPolynomial(timeShifted, GShifted)

maxDiv = findMaxDivergencePeaks(timeShifted, expTuple=expTuple, polyTuple=polyTuple)

rCRTTuple, maxDiv = fitRCRT(timeShifted, GShifted, maxDiv)

expParams, expStdDev = expTuple
polyParams, polyStdDev = polyTuple
rCRTParams, rCRTStdDev = rCRTTuple

funcParams = {"exponential": expParams, "polynomial": polyParams, "rCRT": rCRTParams}
