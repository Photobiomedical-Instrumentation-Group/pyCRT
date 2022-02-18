# Quick script to read the test file and populate the environment with useful variables.
# Only meant to be used for testing the other modules on ipython.

import numpy as np

arq = np.load("_testData.npz")

timeScdsArr = arq["timeScds"]
avgIntenArr = arq["avgInten"]
B, G, R = avgIntenArr[:, 0], avgIntenArr[:, 1], avgIntenArr[:, 2]
