"""Tests for the simpleUI module"""

from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
from pyCRT import PCRT
from pyCRT.videoReading import (estimateVideoFpsFromTimestamps, getFrameCount,
                                videoCapture)


class TestPCRT:
    # {{{
    """Tests for the PCRT class"""

    baseDir = Path(__file__).resolve().parent
    videoPath = baseDir / "data" / "P4CR4.wmv"
    npzPath = baseDir / "data" / "test.npz"

    roi = (436, 358, 270, 130)
    expectedPCRTExclusionMethod = {
        "first that works": (4.600165280378453, 0.19374687901749468),
        "first positive peak": (4.600165280378453, 0.19374687901749468),
        "best fit": (3.00277714177836, 0.05735139417740891),
        "strict": (None, None),  # RuntimeException expected
    }

    expectedPCRT, expectedUnc = expectedPCRTExclusionMethod[
        "first positive peak"
    ]
    expectedString = "4.60 ± 0.19 s (4.21%)"

    expectedPCRTRescaleFactors = {
        0.25: (5.216837599964007, 0.266823469797725),
        0.5: (5.008278833876177, 0.23840868811616253),
        0.75: (5.0276541887059905, 0.2258960764339057),
    }

    rescaleFactorsROIs = {
        0.25: (105, 78, 76, 50),
        0.5: (212, 165, 145, 90),
        0.75: (318, 251, 229, 143),
    }

    testFPS = (20, 50, 100)
    correctFPS = 21.73913043478089
    frameNum = 852

    def testCheckTestVideo(self):
        # {{{
        """Check if test video exists."""
        assert self.videoPath.exists()

    # }}}

    def testNoDisplay(self):
        # {{{
        """
        Test that PCRT.fromVideoFile works when video display and live plotting
        are disabled, and that the calculated pCRT matches the expected values.
        """
        pcrt = PCRT.fromVideoFile(
            str(self.videoPath),
            roi=self.roi,
            displayVideo=False,
            livePlot=False,
        )
        pcrtValue, pcrtUnc = pcrt.pCRT
        assert np.isclose(pcrtValue, self.expectedPCRT)
        assert np.isclose(pcrtUnc, self.expectedUnc)

    # }}}

    def testDisplayVideo(self):
        # {{{
        """
        Test that video display and live plotting work by creating a PCRT
        instance with display enabled, and visually confirming that the plots
        appear as expected.
        """
        pcrt = PCRT.fromVideoFile(
            str(self.videoPath), roi=self.roi, displayVideo=True, livePlot=True
        )
        pcrtValue, pcrtUnc = pcrt.pCRT
        assert np.isclose(pcrtValue, self.expectedPCRT)
        assert np.isclose(pcrtUnc, self.expectedUnc)

        answer = "banana"
        while answer not in ("y", "n"):
            pcrt.showAvgIntensPlot()
            answer = input("Did it look alright? [y/n] ")

        assert answer != "n"

        answer = "banana"
        while answer not in ("y", "n"):
            pcrt.showPCRTPlot()
            answer = input("Did it look alright? [y/n] ")

        assert answer != "n"

    # }}}

    def testSavingLoading(self):
        # {{{
        """
        Test that a PCRT object can be saved to file and reloaded correctly,
        ensuring that time arrays, intensity arrays, and pCRT values are
        preserved.
        """
        pcrtOriginal = PCRT.fromVideoFile(
            str(self.videoPath),
            roi=self.roi,
            displayVideo=False,
            livePlot=False,
        )

        pcrtOriginal.save(str(self.npzPath))

        pcrtLoaded = PCRT.fromArchive(str(self.npzPath))

        assert np.allclose(pcrtLoaded.avgIntensArr, pcrtOriginal.avgIntensArr)
        assert np.allclose(pcrtLoaded.timeScdsArr, pcrtOriginal.timeScdsArr)
        assert np.allclose(pcrtLoaded.pCRT, pcrtOriginal.pCRT)

        if self.npzPath.exists():
            self.npzPath.unlink()

    # }}}

    def testExclusionMethods(self):
        # {{{
        """
        Test that different exclusion methods ('first that works', 'best fit',
        and 'strict') produce the expected pCRT results or expected failure.
        """
        pcrtFirstThatWorks = PCRT.fromVideoFile(
            str(self.videoPath),
            exclusionMethod="first that works",
            roi=self.roi,
            displayVideo=False,
            livePlot=False,
        )
        assert np.allclose(
            pcrtFirstThatWorks.pCRT,
            self.expectedPCRTExclusionMethod["first that works"],
        )

        pcrtFirstPositivePeak = PCRT.fromVideoFile(
            str(self.videoPath),
            exclusionMethod="first positive peak",
            roi=self.roi,
            displayVideo=False,
            livePlot=False,
        )
        assert np.allclose(
            pcrtFirstPositivePeak.pCRT,
            self.expectedPCRTExclusionMethod["first positive peak"],
        )

        pcrtBestFit = PCRT.fromVideoFile(
            str(self.videoPath),
            exclusionMethod="best fit",
            roi=self.roi,
            displayVideo=False,
            livePlot=False,
        )
        assert np.allclose(
            pcrtBestFit.pCRT,
            self.expectedPCRTExclusionMethod["best fit"],
        )

        with pytest.raises(RuntimeError):
            pcrtStrict = PCRT.fromVideoFile(
                str(self.videoPath),
                exclusionMethod="strict",
                roi=self.roi,
                displayVideo=False,
                livePlot=False,
            )
            assert np.allclose(
                pcrtStrict.pCRT,
                self.expectedPCRTExclusionMethod["strict"],
            )
            # }}}

    def testString(self):
        # {{{
        pcrt = PCRT.fromVideoFile(
            str(self.videoPath),
            roi=self.roi,
            displayVideo=False,
            livePlot=False,
        )
        assert str(pcrt) == self.expectedString

    # }}}

    def testRescaleFactor(self):
        # {{{
        for rescaleFactor, ROI in self.rescaleFactorsROIs.items():
            pcrt = PCRT.fromVideoFile(
                str(self.videoPath),
                displayVideo=False,
                livePlot=False,
                rescaleFactor=rescaleFactor,
                roi=ROI,
                exclusionMethod="first positive peak",
            )
            pcrtValue, pcrtUnc = pcrt.pCRT
            expectedValue, expectedUnc = self.expectedPCRTRescaleFactors[
                rescaleFactor
            ]
            assert np.isclose(pcrtValue, expectedValue) and np.isclose(
                pcrtUnc, expectedUnc
            )
        # }}}

    def testEstimateFPS(self):
        # {{{
        with videoCapture(str(self.videoPath), None) as cap:
            estimatedFPS = estimateVideoFpsFromTimestamps(cap, 500)
        assert estimatedFPS == self.correctFPS

    # }}}

    def testFrameCount(self):
        # {{{
        with videoCapture(str(self.videoPath), None) as cap:
            frameCount = getFrameCount(cap, forceFallback=True)
        assert frameCount == self.frameNum

    # }}}

    def testPlaybackFPS(self):
        # {{{
        with videoCapture(str(self.videoPath), None) as cap:
            frameCount = getFrameCount(cap, forceFallback=True)
        for FPS in self.testFPS:
            initial_time = perf_counter()
            PCRT.fromVideoFile(
                str(self.videoPath),
                livePlot=False,
                roi=self.rescaleFactorsROIs[0.25],
                rescaleFactor=0.25,
                playbackFPS = FPS,
            )
            timePassed = perf_counter() - initial_time
            realFPS = frameCount / timePassed
            print(f"Tested FPS: {FPS}")
            print(f"Measured FPS: {round(realFPS, 2)}")
            answer = "banana"
            while answer not in ("y", "n"):
                answer = input("Did it look alright? [y/n] ")
            assert answer != "n"
    # }}}


# }}}
