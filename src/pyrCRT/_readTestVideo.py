from simpleUI import RCRT
import numpy as np

videoPath = "/home/eduardo/Code/Python/pyrCRT-tests/test data/P1-CR1.mp4"
roi = (460, 325, 236, 161)

rcrt = RCRT.fromVideoFile(
    videoPath,
    roi=roi,
    displayVideo=False,
    exclusionCriteria=np.inf,
    exclusionMethod="first that works",
)
