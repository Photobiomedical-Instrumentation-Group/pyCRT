# Quick script to read the test file and populate the environment with useful variables.
# Only meant to be used for testing the other modules on ipython.

import numpy as np
from simpleUI import RCRT

arq = np.load("_testData.npz")

fullTimeScds = arq["fullTimeScds"]
fullAvgIntens = arq["fullAvgIntens"]

rcrt = RCRT(fullTimeScds, fullAvgIntens, exclusionMethod="best fit")
