"""
Converts videos into npz files of channelFullAvgIntens of the G channel
"""

from pathlib import Path

import numpy as np
from pyCRT import PCRT

testDataDir = Path("data")
npzDataDir = Path("npzData")

for entry in testDataDir.iterdir():
    if not entry.is_file() or entry.stem.split("_")[0] == "ROI":
        continue

    destPath = npzDataDir / f"{entry.stem}.npz"

    pcrt = PCRT.fromVideoFile(
        str(entry),
        roi="all",
        displayVideo=False,
        livePlot=False,
        exclusionCriteria=np.inf,
        exclusionMethod="first that works",
    )
    fromIndex, toIndex, _ = pcrt.slice.indices(len(pcrt.fullTimeScdsArr))

    np.savez(
        str(destPath),
        channelFullAvgIntens=pcrt.channelFullAvgIntens,
        timeScdsArr=pcrt.fullTimeScdsArr,
        fromIndex=fromIndex,
        toIndex=toIndex,
    )

    print(destPath)
