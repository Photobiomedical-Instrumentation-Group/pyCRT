# Usage

## Installation

Install pyrCRT through pip, preferably in a [virtual
environment](https://docs.python.org/3/tutorial/venv.html):

```{code-block} console
pip install pyrCRT
```

## Basic usage

PyrCRT's main user interface is its RCRT class, which represents a single
measurement. To calculate the rCRT from a video file, for example, the
following code is all that is strictly necessary:

```{code-block} python
from pyrCRT import RCRT

# Path to the video in the file system
filePath = "video.wmv"

rcrt = RCRT.fromVideoFile(filePath)
```

At this point a window with video playback will appear. In order to select a
_region of interest_ (ROI), press the space bar at any moment and the video
will pause, allowing you to drag a square around the desired ROI.

```{image} ROIselection.jpg
:scale: 50 %
:align: center
```

Press the space bar again to confirm the selection and the video will resume.
You'll notice a 4 elements tuple printed on the terminal after confirming the
ROI. This tuple specifies the ROI, and can be passed to `RCRT.fromVideoFile`
method via the `roi` keyword argument to avoid having to manually select the
ROI next time.

After the video ends, the window closes automatically and pyrCRT attempts to
calculate the rCRT. The rCRT and its 0.95 confidence interval uncertainty are
attributes of the rCRT instance created through the `RCRT.fromVideoFile`
method, which is now stored at the `rcrt` variable:


```{code-block} python
>>> print(rcrt)
1.68±3.27%
>>>print(rcrt.rCRT) # the second element is the absolute uncertainty
(1.6790549329488764, 0.054861676836145)
```

To view graphs of the pixel's average intensities inside the ROI for each
channel, and of the curve fits performed on the channel with which the rCRT is
calculated (G, by default), use the following:

```{code-block} python
rcrt.showAvgIntensPlot()
rcrt.showRCRTPlot()
```

This will produce the following two graphs, respectively:

```{image} plots.jpg
:width: 100 %
:align: center
```

As can be seen by the graph on the right, pyrCRT's default behaviour is to use
the first peak of the polynomial - exponential function as the critical time.
This behaviour can be changed by changing the value of the `exclusionMethod`
keyword argument 

This simple tutorial shows only the bare minimum to use pyrCRT. For a complete
reference of pyrCRT's functionalities and options, consult the {doc}`
API reference <pyrCRT>`, especially the simpleUI.RCRT class.
