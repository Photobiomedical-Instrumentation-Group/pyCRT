"""
Video reading functionality (pyrCRT.videoReading)

This module contains everything directly related to extracting lightly processed data
from video files, in particular obtaining the arrays with each frame's time within the
video and the average intensities array calculated within a region of interest of each
frame.
"""


import os

import cv2 as cv  # pylint: disable=import-error
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import Tuple
from contextlib import contextmanager

# Type aliases for commonly used types
# {{{
# Standard ROI tuple used by OpenCV
roiTuple = Tuple[int, int, int, int]

# Tuples of two numpy arrays, typically an array of the timestamp for each frame and an
# array of average intensities within a given ROI
arrayTuple = Tuple[np.ndarray, np.ndarray]
# }}}

def readVideo(videoPath: str, roi: Optional[roiType] = None) -> arrayTuple:
    """
    Extracts the time in seconds of each frame and the average pixel intensities of each
    channel within the region of interest (ROI).

    Parameters
    ----------
    videoPath : str
        Path to the video file in the filesystem.
    roi : Tuple[int, int, int, int] or None
        The region of interest, inside which the average of each pixel will be computed.
        This tuple must contain 4 integers: (x, y, length_x, and length_y), where x and
        y are the coordinates of the rectangle's center.

    Returns
    -------
    timeScdsArr : 1D np.ndarray
        Time in seconds of each frame in the video.
    avgIntenArr : 2D np.ndarray
        Average pixel intensity inside the region of interest with respect to time. It
        is an array with shape (len(timeScdsArr), 3), wherein each element is an array
        with the average intensity of the B, G and R (respectively) channels in that
        instant.

    Raises
    ------
    RuntimeError
        


    Notes
    -----
        This function (this whole module, actually) has only been tested with wmv and
        mp4 files, with WMV2, and DIVX and MP4V codecs respectively.

