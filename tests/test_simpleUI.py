"""Tests for the simpleUI module"""

from pathlib import Path

import numpy as np
import pytest
from pyCRT import PCRT


class TestPCRT:
    # {{{
    """Tests for the PCRT class"""

    baseDir = Path(__file__).resolve().parent
    videoPath = baseDir / "P4CR4.wmv"
    npzPath = baseDir / "test.npz"

    roi = (436, 358, 270, 130)
    expectedPCRTExclusionMethod = {
        "first that works": (4.600165280378453, 0.19374687901749468),
        "best fit": (3.00277714177836, 0.05735139417740891),
        "strict": (None, None),  # RuntimeException expected
    }

    expectedPCRT, expectedUnc = expectedPCRTExclusionMethod["first that works"]

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


# }}}
