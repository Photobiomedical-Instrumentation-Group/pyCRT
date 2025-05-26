"""
A simplified object-oriented interface for pyCRT (pyCRT.simpleUI)

This module provides the PCRT class, which is meant to be the simplest way to
use pyCRT's functions distributed among it's other modules.
"""

from __future__ import annotations

from os.path import isfile
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .arrayOperations import (
    findValueIndex,
    minMaxNormalize,
    sliceByTime,
    sliceFromLocalMax,
    sliceFromMaxToEnd,
    subtractMinimum,
)
from .arrayPlotting import (
    saveAvgIntensPlot,
    savePCRTPlot,
    showAvgIntensPlot,
    showPCRTPlot,
)
from .curveFitting import (
    calcPCRT,
    calculateRelativeUncertainty,
    fitExponential,
    fitPolynomial,
    pCRTFromParameters,
    fit_crt10010exp,
    
    fit_crt10010,
    fit_CRT9010,
    
    fitECRTKShinozaki,
    fitECRT
)

from .videoReading import readVideo

# Type aliases for commonly used types
# {{{
# Array of arbitraty size with float elements.
Array = NDArray[np.float64]

# Tuples of two numpy arrays, typically one array of times and one of average
# intensities or of optimized parameters and their standard deviations.
ArrayTuple = tuple[Array, Array]

# Type for something that can be used as the parameters for some curve-fitting
# function
ParameterSequence = Union[Sequence[float], Array]

# The return type for functions that fit curves. The first element is the
# optimized parameters and the second their standard deviations
FitParametersTuple = tuple[ParameterSequence, ParameterSequence]

Real = Union[float, int, np.float64, int]

# This accounts for the fact that int doesn't inherit from int
Integer = Union[int, int]

FigAxTuple = tuple[Figure, Axes]

# Standard ROI tuple used by OpenCV
RoiTuple = tuple[int, int, int, int]

# Either a RoiTuple, or "all"
RoiType = Union[RoiTuple, str]
# }}}

# Constants
CHANNEL_INDICES_DICT = {"b": 0, "g": 1, "r": 2}


class PCRT:
    # {{{
    """
    Representation of a pCRT measurement. Aside from the pCRT value itself, it
    also stores all the information that went into calculating the pCRT, as
    well as functions related to this calculation and storing the results in a
    file. This class is meant to be the easiest way to use pyCRT.
    """

    # Init methods{{{
    def __init__(
        self,
        fullTimeScdsArr: Array,
        channelsAvgIntensArr: Array,
        channel: str = "g",
        fromTime: Optional[Real] = None,
        toTime: Optional[Real] = None,
        sliceMethod: str = "from local max",
        funcParamsTuples: Optional[dict[str, FitParametersTuple]] = None,
        initialGuesses: Optional[dict[str, ParameterSequence]] = None,
        criticalTime: Optional[float] = None,
        exclusionCriteria: float = 0.12,
        exclusionMethod: str = "best fit",
    ):
        # {{{
        # {{{
        """
        Initializes the PCRT instance with all the parameters that are
        necessary for calculating the pCRT. This method has many parameters but
        only fullTimeScdsArr and channelsAvgIntensArr are required, though it
        is important to know about each parameter.

        Parameters
        ----------
        fullTimeScdsArr : np.ndarray of float
            An array of time instants measured in seconds, with respect to the
            start of the recording.

        channelsAvgIntensArr : np.ndarray of float
            The array of average channel intensities over the pixels in a
            certain region of interest (ROI). Each line is an array of the
            average intensities for each channel (in the order of BGR).

            This, and fullTimeScdsArr are typically the outputs of
            videoReading.readVideo.

        channel : str, default='g'
            The channel that will be used to calculate the pCRT. Can be 'b',
            'g' or 'r'. This argument specifies which column of
            channelsAvgIntensArr will be stored as the channelFullAvgIntens
            property of this instance (see PCRT.channelFullAvgIntens).

        fromTime, toTime : float or None, default=None
            The elements of fullTimeScdsArr from and to which the pCRT
            phenomenon actually takes place. This is the interval of
            channelFullAvgIntens. If None, this interval will be from the first
            and to the last elements of fullTimeScdsArr respectively. The
            rcrt.fromTime and rcrt.toTime attributes won't be the same as these
            parameters after initialization, though, that depends on the
            sliceMethod (see below).

        sliceMethod : str, default='from local max'
            Which way the slice attribute is to be calculated from the fromTime
            and toTime parameters. See PCRT.setSlice for details.

        funcParamsTuples : dict of str keys and tuples of 2 np.ndarray of float
        as values or None, default=None
            The fitted parameters and their standard deviations of the
            exponential, polynomial and pCRT exponential functions can be
            specified via this dictionary if they have been calculated
            beforehand. The keys are 'exponential', 'polynomial' and 'pCRT'
            respectively. A function that doesn't have its key in this
            dictionary will be fitted during the initialization of the PCRT
            instance.

        initialGuesses : dict of str keys and sequence of float values or None,
        default=None
            The initial guesses for each function. The valid keys are the same
            as of funcParamsTuples. For each function whose key is not in this
            dict, the default initial guesses (defined in the curveFitting
            module) will be used.

        criticalTime : float or None, default=None
            The critical time used for the pCRT exponential function fitting.
            If None, it will be calculated by calcPCRT from the time and
            intensity arrays or from the fitted polynomial and exponential
            function parameters. (see curveFitting.calcPCRT)

        exclusionCriteria : float, default=0.12
            The maximum relative uncertainty a pCRT measurement can have and
            not be rejected. If all fits on the criticalTime candidates fail
            this criteria, a RuntimeError will be raised. See
            curveFitting.calcPCRT for more information.

        relativeUncertainty : float
            The pCRT's relative uncertainty.

        exclusionMethod : str, default='best fit'
            Which criticalTime and its associated fitted pCRT parameters and
            standard deviations are to be returned by calcPCRT. Possible values
            are 'best fit', 'strict' and 'first that works' (consult the
            documentation for the calcPCRTBestFit, calcPCRTStrict and
            calcPCRTFirstThatWorks functions from the curveFitting module for a
            description of the effect of these possible values).
        """
        # }}}
        self._crt_10010 = None
        self._crt_9010 = None
        self.k_10 = None
        self._AmplitudeAC = None
        self._AmplitudeDC = None
        
        self._crt_10010exp = None
        self._incer_10010exp = None
        self._uncertainty_crt_10010=None
        self._time10 = None
        
        
        
         # Input parameters of CRT function with exponential 
        
        # Input parameters of CRT function with exponential until 10% decay
        self._eCRT10010=None
        self._eCRTtime10=None
        
        # Input parameters of CRT exponential
        self._eCRT=None
        self._UncertaintyECRT=None
        
        self.fullTimeScdsArr = fullTimeScdsArr
        self.channelsAvgIntensArr = channelsAvgIntensArr
        self.channel = channel.strip().lower()
        self.setSlice(fromTime, toTime, sliceMethod)

        if initialGuesses is None:
            self.initialGuesses = {}
        else:
            self.initialGuesses = initialGuesses

        if funcParamsTuples is None:
            funcParamsTuples = {}

        if "exponential" in funcParamsTuples:
            self.expParams, self.expStdDev = funcParamsTuples["exponential"]
        else:
            self.expParams, self.expStdDev = fitExponential(
                self.timeScdsArr,
                self.avgIntensArr,
                self.initialGuesses.get("exponential", None),
            )

        if "polynomial" in funcParamsTuples:
            self.polyParams, self.polyStdDev = funcParamsTuples["polynomial"]
        else:
            self.polyParams, self.polyStdDev = fitPolynomial(
                self.timeScdsArr,
                self.avgIntensArr,
                self.initialGuesses.get("polynomial", None),
            )

        if "pCRT" in funcParamsTuples and criticalTime is not None:
            self.pCRTParams, self.pCRTStdDev = funcParamsTuples["pCRT"]

        else:
            (
                self.pCRTParams,
                self.pCRTStdDev,
            ), self.criticalTime = self.calcPCRT(
                criticalTime,
                self.initialGuesses.get("pCRT", None),
                exclusionMethod,
                exclusionCriteria,
            )
    
    def calculate_crt_9010(self):
        try:
            self.crt_9010, self.k_10, self.AmplitudeAC, self.AmplitudeDC = fit_CRT9010(self.timeScdsArr, self.avgIntensArr)
        except Exception as e:
            print(f"Erro durante o cálculo do CRT 9010: {e}")
            self.crt_9010 = None
            self.k_10 = None
            self.AmplitudeAC = None
            self.AmplitudeDC = None

        # Verifique se a função retornou valores válidos
        if self.crt_9010 is None or self.k_10 is None :
            print("Erro: A função fit_CRT9010 retornou None.")

    # calculate CRT100-10
    def calculate_crt_10010(self):
        self.crt_10010= fit_crt10010(self.timeScdsArr, self.avgIntensArr,self.k_10)

    

    def calculate_crt_10010exp(self) -> float:
        try:
            # Chamar o cálculo do CRT10010 exp
            #resultado = fit_crt10010exp(self.timeScdsArr, self.avgIntensArr, self.k_10)
            crt_10010exp, uncertainty_crt_10010,time10 = fit_crt10010exp(self.timeScdsArr, self.avgIntensArr, self.k_10)
            
            # Armazenar o valor calculado internamente
            self._crt_10010exp = crt_10010exp
            self._uncertainty_crt_10010 = uncertainty_crt_10010
            self._time10 = time10

            # Retornar o valor calculado
            return self._crt_10010exp,self._uncertainty_crt_10010, self._time10
        except Exception as e:
            print(f"Erro ao calcular crt_10010exp: {e}")
            return None  # Retornar None em caso de erro

    #{{{ Analise temporal
    # Metodo K.Shinozaki exponencial como modelo para adquirir os valores de tempo de 10% e 100% da intensidade máxima  
    def calculatetCRT10(self) -> float:
        try:
            
            BRT,tcBRT,rrBRT= fitECRTKShinozaki(self.timeScdsArr, self.avgIntensArr)
            
            self._BRT = BRT
            self._tcBRT = tcBRT
            self._rrBRT = rrBRT

            # Retornar o valor calculado
            return self._BRT, self._tcBRT, self._rrBRT
        except Exception as e:
            print(f"Erro ao calcular tCRT10: {e}")
            return None  

    # função que retorna os valores da função fitECRT - exponencial
    def calculateECRT(self) -> float:
        try:
            
            eCRT,UncertaintyECRT = fitECRT(self.timeScdsArr, self.avgIntensArr)
            
            self._eCRT = eCRT
            self._UncertaintyECRT = UncertaintyECRT

            # Retornar o valor calculado
            return self._eCRT, self._UncertaintyECRT
        except Exception as e:
            print(f"Erro ao usar o método eCRT: {e}")
            return None  


    @classmethod
    def fromVideoFile(
        cls,
        videoPath: str,
        roi: Optional[RoiType] = None,
        displayVideo: bool = True,
        livePlot: bool = False,
        rescaleFactor: Real = 1.0,
        waitKeyTime: int = 1,
        **kwargs: Any,
    ) -> PCRT:
        # {{{
        # {{{
        """
        Creates the fullTimeScdsArr and channelsAvgIntensArr arrays from a
        video file and initializes the PCRT instance with these arrays, and
        additional arguments passed to this function as kwargs.

        Parameters
        ----------
        roi : Tuple[int, int, int, int] or "all"
            The region of interest, inside which the average of each pixel will
            be computed.  This tuple must contain 4 integers: (x, y, length_x,
            and length_y), where x and y are the coordinates of the rectangle's
            center. If roi == "all", then the average will be computed on the
            entire frame. You can also select the ROI at any time during the
            video by pressing the space bar and dragging the square around the
            desired region.

        displayVideo : bool, default=True
            Whether or not to display the video while it is being read. This
            must be set to True if no ROI is specified, so the ROI can be
            manually selected by pressing the spacebar during the video
            exhibition.

        livePlot : bool, default=True
            Whether to display a live plot of the pixel intensity over time as
            the video is read. This can be useful for real-time monitoring of
            signal changes within the selected region of interest (ROI). The
            live plot will update dynamically with each processed frame. For
            this feature to work, a valid ROI must be specified either
            beforehand or manually selected during playback. If False, no plot
            will be shown during video capture.

        rescaleFactor : real number, optional
            Factor by which each frame will be scaled. This can help reduce the
            load on the hardware and speed up computation. By default the video
            won't be scaled.

        waitKeyTime : int, optional
            How many milliseconds to wait for user input between each frame.
            The default value is 1, so on most machines the video will appear
            "sped up" relative to it being played on a regular video player.
            See cv2.waitKey for more information.

        kwargs : dict of str keys and any value
            These additional arguments will be passed to this class's __init__
            method.

        See Also
        --------
        videoReading.readVideo :
            Basically all this function does is call videoReading.readVideo
            with the arguments passed to this function, so refer to that
            function's docstring for more information.
        """
        # }}}
        if not isfile(videoPath):
            raise ValueError(
                f"No file exists on path {videoPath}. If you wanted to read "
                "data from a capture device such as a webcam, see "
                "simpleUI.fromCaptureDevice."
            )

        fullTimeScdsArr, channelsAvgIntensArr = readVideo(
            videoPath,
            roi=roi,
            displayVideo=displayVideo,
            rescaleFactor=rescaleFactor,
            waitKeyTime=waitKeyTime,
            livePlot=livePlot,
        )

        return cls(fullTimeScdsArr, channelsAvgIntensArr, **kwargs)

    # }}}

    @classmethod
    def fromCaptureDevice(
        cls,
        capDeviceIndex: int,
        roi: Optional[RoiType] = None,
        livePlot: bool = True,
        cameraResolution: Optional[Tuple[int, int]] = None,
        recordingPath: Optional[str] = None,
        codecFourcc: str = "mp4v",
        recordingFps: float = 30.0,
        **kwargs: Any,
    ) -> PCRT:
        # {{{
        # {{{
        """
        Creates the fullTimeScdsArr and channelsAvgIntensArr arrays from video
        read from a video capture device, and initializes the PCRT instance
        with these arrays and additional arguments passed to this function as
        kwargs.

        Parameters
        ----------
        roi : Tuple[int, int, int, int] or "all"
            The region of interest, inside which the average of each pixel will
            be computed.  This tuple must contain 4 integers: (x, y, length_x,
            and length_y), where x and y are the coordinates of the rectangle's
            center. If roi == "all", then the average will be computed on the
            entire frame. You can also select the ROI at any time during the
            video by pressing the space bar and dragging the square around the
            desired region.

        livePlot : bool, default=True
            Whether to display a live plot of the pixel intensity over time as
            the video is read. This can be useful for real-time monitoring of
            signal changes within the selected region of interest (ROI). The
            live plot will update dynamically with each processed frame. For
            this feature to work, a valid ROI must be specified either
            beforehand or manually selected during playback. If False, no plot
            will be shown during video capture.

        cameraResolution : tuple of 2 ints, default=None
            Used to optionally change the camera resolution before handing over
            the VideoCapture instance. If reading from a video file, it does
            nothing.

        recordingPath : str, default=None
            The path (with the extension!) in the filesystem wherein to save
            the recording.  If false, it won't record the video.

        codecFourcc : str, default='mp4v'
            The fourcc identifier for the video codec to be used for the
            recording. Refer to www.fourcc.org/codecs.php for a list of
            possible codes.

        recordingFps : float, default=None
            The FPS (frames per second) for the recording, which doesn't need
            to correspond to the FPS of the camera.

        kwargs : dict of str keys and any values
            These additional arguments will be passed to this class's __init__
            method.

        See Also
        --------
        videoReading.readVideo :
            Basically all this function does is call videoReading.readVideo
            with the arguments passed to this function, so refer to that
            function's docstring for more information.
        """
        # }}}

        if not isinstance(capDeviceIndex, int):
            raise TypeError(
                f"Invalide value of {capDeviceIndex} for the capDeviceIndex "
                "argument device. To list the available capture devices, use "
                "videoReading.listCaptureDevices. If you wanted to read data "
                "from a video file, see simpleUI.fromVideoFile."
            )

        fullTimeScdsArr, channelsAvgIntensArr = readVideo(
            capDeviceIndex,
            roi=roi,
            cameraResolution=cameraResolution,
            recordingPath=recordingPath,
            codecFourcc=codecFourcc,
            recordingFps=recordingFps,
            rescaleFactor=1.0,
            livePlot=livePlot,
        )
        return cls(fullTimeScdsArr, channelsAvgIntensArr, **kwargs)

    # }}}

    @classmethod
    def fromArchive(cls, filePath: str) -> PCRT:
        # {{{
        # {{{
        """
        Creates an PCRT instance from the data stored in a file created by the
        PCRT.save method.

        Parameters
        ----------
        filePath : str
            The path to the file in the file system. It must be a numpy npz
            archive containing fullTimeScdsArr, channelsAvgIntenArr, channel,
            fromTime, toTime, expTuple, polyTuple, pCRTTuple and criticalTime
            (see the init method of PCRT for an explanation of what each of
            these are supposed to be).

        See Also
        --------
        PCRT.save :
            Use this method to store the PCRT measurement in a file that is
            retrievable by this method.
        """
        # }}}

        archive = np.load(filePath)

        return cls(
            fullTimeScdsArr=archive["fullTimeScdsArr"],
            channelsAvgIntensArr=archive["channelsAvgIntensArr"],
            channel=str(archive["channel"]),
            fromTime=float(archive["fromTime"]),
            toTime=float(archive["toTime"]),
            funcParamsTuples={
                "exponential": tuple(archive["expTuple"]),  # type: ignore
                "polynomial": tuple(archive["polyTuple"]),  # type: ignore
                "pCRT": tuple(archive["pCRTTuple"]),  # type: ignore
            },
            criticalTime=float(archive["criticalTime"]),
            sliceMethod="from local max",
        )
        # }}}}}}

    # Instance methods{{{
    def save(self, filePath: str) -> None:
        # {{{
        # {{{
        """
        Saves the relevant attributes of this PCRT instance in a numpy npz file
        on the specified file path. To retrieve the data in this file later,
        use PCRT.fromArchive.
        """
        # }}}

        np.savez(
            filePath,
            fullTimeScdsArr=self.fullTimeScdsArr,
            channelsAvgIntensArr=self.channelsAvgIntensArr,
            channel=self.channel,
            fromTime=self.fromTime,
            toTime=self.toTime,
            expTuple=np.array(self.expTuple),
            polyTuple=np.array(self.polyTuple),
            pCRTTuple=np.array(self.pCRTTuple),
            criticalTime=self.criticalTime,
        )

    # }}}

    def setSlice(
        self,
        fromTime: Optional[Real] = None,
        toTime: Optional[Real] = None,
        sliceMethod: str = "from local max",
    ) -> None:
        # {{{
        # {{{
        """
        Sets the slice object that will be used to slice the
        channelFullAvgIntens and fullTimeScdsArr arrays to produce avgIntensArr
        and timeScdsArr respectively. This slice is supposed to indicate only
        the region wherein the CRT phenomenon takes place. This slice is used
        for PCRT.timeScdsArr and PCRT.avgIntensArr.


        Parameters
        ----------
        fromTime, toTime : float or None, default=None
            The elements of fullTimeScdsArr from and to which the pCRT
            phenomenon actually takes place. This is the interval of
            channelFullAvgIntens. If None, this interval will be from the first
            and to the last elements of fullTimeScdsArr respectively. The
            rcrt.fromTime and rcrt.toTime attributes won't be the same as these
            parameters after initialization, though, that depends on the
            sliceMethod (see below).

        sliceMethod : str, default='from local max'
            Which way the slice attribute is to be calculated from the fromTime
            and toTime parameters. The possible values are:

            'from max': the slice will be calculated from the absolute maximum
            of channelFullAvgIntens up to the array's end. The fromTime and
            toTime arguments are not necessary for this, and in fact this is
            this function's behaviour when they are both None.

            'by time': the slice will start on the first element of timeScdsArr
            that is greater than or equal to fromTime and end on the first
            element that is greater than or equal to toTime.

            'from local max': the slice will start on the maximum of
            channelFullAvgIntens between the indexes of fullTimeScdsArr that
            correspond to fromTime and toTime, and end on toTime.
        """
        # }}}

        sliceMethod = sliceMethod.strip().lower()

        if sliceMethod == "from max" or (fromTime, toTime) == (None, None):
            self.slice = sliceFromMaxToEnd(self.channelFullAvgIntens)
        elif sliceMethod == "from local max":
            self.slice = sliceFromLocalMax(
                self.fullTimeScdsArr,
                self.channelFullAvgIntens,
                fromTime,
                toTime,
            )
        elif sliceMethod == "by time":
            self.slice = sliceByTime(self.fullTimeScdsArr, fromTime, toTime)
        else:
            raise ValueError(
                f"Invalid value of '{sliceMethod}' passed as sliceMethod. "
                "Valid values: 'from local max', 'from max' and 'by time'"
            )

        fromIndex, toIndex, _ = self.slice.indices(len(self.fullTimeScdsArr))
        self.fromTime, self.toTime = (
            self.fullTimeScdsArr[fromIndex],
            self.fullTimeScdsArr[toIndex],
        )

    # }}}

    def calcPCRT(
        self,
        criticalTime: Optional[float] = None,
        pCRTInitialGuesses: Optional[ParameterSequence] = None,
        exclusionMethod: str = "best fit",
        exclusionCriteria: float = np.inf,
    ) -> Tuple[ArrayTuple, float]:
        # {{{
        # {{{
        """
        Simply returns the output of curveFitting.calcPCRT called with the
        instance's attributes and this function's parameters as its arguments.
        See curveFitting.calcPCRT for a detailed explantion.
        """
        # }}}

        return calcPCRT(
            self.timeScdsArr,
            self.avgIntensArr,
            criticalTime,
            self.expTuple,
            self.polyTuple,
            pCRTInitialGuesses,
            exclusionMethod,
            exclusionCriteria,
        )

    # }}}

    def showAvgIntensPlot(self) -> None:
        # {{{
        # {{{
        """
        Shows the plot of average intensities for every channel in function of
        the timestamp of each frame in the entire video. See
        arrayPlotting.makeAvgIntensPlot and arrayPlotting.showAvgIntensPlot.
        """
        # }}}
        showAvgIntensPlot(self.fullTimeScdsArr, self.channelsAvgIntensArr)

    # }}}

    def saveAvgIntensPlot(self, figPath: str) -> None:
        # {{{
        # {{{
        """
        Saves the plot of average intensities for every channel in function of
        the timestamp of each frame in the entire video to a file. See
        arrayPlotting.makeAvgIntensPlot and arrayPlotting.saveAvgIntensPlot.
        """
        # }}}
        saveAvgIntensPlot(
            figPath, self.fullTimeScdsArr, self.channelsAvgIntensArr
        )

    # }}}

    def showPCRTPlot(self) -> None:
        # {{{
        # {{{
        """
        Shows the plot of normalized average intensities for the channel
        specified in the initialization of the PCRT instance, in function of
        the time since the removal of the pressure from the skin, and the
        fitted functions on this data. This is supposed to show only the CRT
        phenomenon. See arrayPlotting.makePCRTPlot and
        arrayPlotting.showPCRTPlot.
        """
        # }}}
        showPCRTPlot(
            self.timeScdsArr,
            self.avgIntensArr,
            {
                "exponential": self.expTuple,
                "polynomial": self.polyTuple,
                "pCRT": self.pCRTTuple,
            },
            self.criticalTime,
            self.channel,
        )

    # }}}

    def savePCRTPlot(self, figPath: str) -> None:
        # {{{
        """
        Saves the plot of normalized average intensities for the channel
        specified in the initialization of the PCRT instance, in function of
        the time since the removal of the pressure from the skin, and the
        fitted functions on this data. This is supposed to show only the CRT
        phenomenon. See arrayPlotting.makePCRTPlot and
        arrayPlotting.showPCRTPlot.
        """

        savePCRTPlot(
            figPath,
            self.timeScdsArr,
            self.avgIntensArr,
            {
                "exponential": self.expTuple,
                "polynomial": self.polyTuple,
                "pCRT": self.pCRTTuple,
            },
            self.criticalTime,
            self.channel,
        )

    # }}}
    # }}}

    # Several properties, mostly for convenience and organization{{{
    @property
    def B(self) -> Array:
        # {{{
        # {{{
        """
        The average intensities of the B channel, measured from the start of
        the recording.
        """
        # }}}
        return self.channelsAvgIntensArr[:, 0]

    # }}}

    @property
    def G(self) -> Array:
        # {{{
        # {{{
        """
        The average intensities of the G channel, measured from the start of
        the recording.
        """
        # }}}
        return self.channelsAvgIntensArr[:, 1]

    # }}}

    @property
    def R(self) -> Array:
        # {{{
        # {{{
        """
        The average intensities of the R channel, measured from the start of
        the recording.
        """
        # }}}

        return self.channelsAvgIntensArr[:, 2]

    # }}}

    @property
    def channelFullAvgIntens(self) -> Array:
        # {{{
        # {{{
        """
        The average intensities of the channel specified by the 'channel'
        instance attribute, measured from the start of the recording.
        """
        # }}}
        return self.channelsAvgIntensArr[:, CHANNEL_INDICES_DICT[self.channel]]

    # }}}

    @property
    def timeScdsArr(self) -> Array:
        # {{{
        # {{{
        """
        The array of time instants in seconds, measured from the removal of the
        pressure from the skin. This array is supposed to encompass only
        portion of fullTimeScdsArr wherein the CRT phenomenon occurs.
        """
        # }}}
        return subtractMinimum(self.fullTimeScdsArr[self.slice])

    # }}}

    @property
    def avgIntensArr(self) -> Array:
        # {{{
        # {{{
        """
        The array of normalized average intensities of a channel. This array
        goes along with timeScdsArr in that it encompasses only the portion of
        channelFullAvgIntens wherein the CRT phenomenon occurs.
        """
        # }}}
        return minMaxNormalize(self.channelFullAvgIntens[self.slice])

    # }}}

    @property
    def expTuple(self) -> FitParametersTuple:
        # {{{
        # {{{
        """
        The optimized parameters and standard deviations of the exponential
        function fitted on f(timeScdsArr)=avgIntensArr. See
        curveFitting.fitExponential.
        """
        # }}}
        return (self.expParams, self.expStdDev)

    # }}}

    @property
    def polyTuple(self) -> FitParametersTuple:
        # {{{
        # {{{
        """
        The optimized parameters and standard deviations of the polynomial
        function fitted on f(timeScdsArr)=avgIntensArr. See
        curveFitting.fitPolynomial.
        """
        # }}}

        return (self.polyParams, self.polyStdDev)

    # }}}

    @property
    def pCRTTuple(self) -> FitParametersTuple:
        # {{{
        # {{{
        """
        The optimized parameters and standard deviations of the pCRT
        exponential function fitted on f(timeScdsArr)=avgIntensArr. See
        curveFitting.fitPCRT.
        """
        # }}}

        return (self.pCRTParams, self.pCRTStdDev)

    # }}}

    @property
    def pCRT(self) -> Tuple[float, float]:
        # {{{
        # {{{
        """
        The pCRT and its uncertainty with a 95% confidence interval, as
        calculated by curveFitting.calcPCRT.
        """
        # }}}
        return pCRTFromParameters(self.pCRTTuple)

    # }}}

    #{{amplitudes
    
    @property
    def AmplitudeAC(self) -> float:
        """
        Get the value of the CRT 9010 time.

        Returns:
        float: The value of the CRT 9010 time.
        """
        return self._AmplitudeAC

    @AmplitudeAC.setter
    def AmplitudeAC(self, value: float):
        """
        Set the value of the CRT 9010 time.

        Parameters:
            value (float): The new value for the CRT 9010 time.
        """
        self._AmplitudeAC = value

    
    @property
    def AmplitudeDC(self) -> float:
        """
        Get the value of the CRT 9010 time.

        Returns:
        float: The value of the CRT 9010 time.
        """
        return self._AmplitudeDC

    @AmplitudeDC.setter
    def AmplitudeDC(self, value: float):
        """
        Set the value of the CRT 9010 time.

        Parameters:
            value (float): The new value for the CRT 9010 time.
        """
        self._AmplitudeDC = value

            
        
    #}}

    @property
    def crt_9010(self) -> float:
        """
        Get the value of the CRT 9010 time.

        Returns:
        float: The value of the CRT 9010 time.
        """
        return self._crt_9010

    @crt_9010.setter
    def crt_9010(self, value: float):
        """
        Set the value of the CRT 9010 time.

        Parameters:
            value (float): The new value for the CRT 9010 time.
        """
        self._crt_9010 = value

    @property
    def crt_10010(self) -> float:
        """
        Get the value of the CRT 10010 time .
        Returns:
        float: The value of the CRT 10010 time.
        """

        return self._crt_10010

    @crt_10010.setter
    def crt_10010(self, value: float):
            """
            Set the value of the CRT 10010 time .

            Parameters:
                value (float): The new value for the CRT 10010 time .
            """
            self._crt_10010 = value

    @property
    def crt_10010exp(self) -> float:
        """
        Retorna o valor calculado do CRT 10010 exponencial.
        """
        return self._crt_10010exp

    @crt_10010exp.setter
    def crt_10010exp(self, value: float):
        """
        Define o valor para o CRT 10010 exponencial.
        """
        self._crt_10010exp = value

    
    #eCRT = exponencial inteira 
    @property
    def eCRT(self) -> float:
            """
            Retorna o valor calculado do CRT exponencial.
            """
            return self._eCRT

    @eCRT.setter
    def eCRT(self, value: float):
            """
            Define o valor para o CRT  exponencial.
            """
            self._eCRT = value
    
    @property
    def UncertaintyECRT(self) -> float:
            """
            Retorna o valor do erro do CRT calculado pela exponencial 
            """
            return self._UncertaintyECRT

    @UncertaintyECRT.setter
    def UncertaintyECRT(self, value: float):
            """
            Define o valor do erro do CRT calculado pela exponencial 
            """
            self._UncertaintyECRT = value
    
    # Método K.Shinozaki para calcular o CRT
    @property
    def BRT(self) -> float:
            return self._BRT

    @BRT.setter
    def BRT(self, value: float):
            self._BRT= value
    
    
    @property
    def tcBRT(self) -> float:
            return self._tcBRT

    @tcBRT.setter
    def tcBRT(self, value: float):
            self._tcBRT = value
    
    
    @property
    def rrBRT(self) -> float:
            return self._rrBRT

    @rrBRT.setter
    def rrBRT(self, value: float):
            self._rrBRT = value
    

    @property
    def criticalTime(self) -> float:
        # {{{
        # {{{
        """
        The critical time used for the pCRT calculation, either set via keyword
        argument during the instance's initialization or calculated by
        curveFitting.calcPCRT.
        """
        # }}}
        self.maxDiv: int
        return self.timeScdsArr[self.maxDiv]

    # }}}

    @criticalTime.setter
    def criticalTime(self, value: Real) -> None:
        # {{{
        """
        Setter for the criticalTime property.
        """

        self.maxDiv = int(findValueIndex(self.timeScdsArr, value))

    # }}}

    @property
    def relativeUncertainty(self) -> np.float64:
        # {{{
        # {{{
        """
        The relative uncertainty (with a 95% confidence interval) for the pCRT
        measurement, on the scale from 0 to 1.
        """
        # }}}
        return calculateRelativeUncertainty(self.pCRTTuple)

    # }}}
    # }}}

    # Magic methods{{{
    def __str__(self) -> str:
        # {{{
        # {{{
        """
        String representation of the pCRT measurement. Gives a fancy string
        with the pCRT and relative uncertainty.
        """
        # }}}
        return f"{self.pCRT[0]:.2f}±{100*self.relativeUncertainty:.2f}%"

    # }}}

    def __repr__(self) -> str:
        # {{{
        # {{{
        """
        Representation of the pCRT measurement. Just returns PCRT.pCRT.
        """
        # }}}
        return str(self.pCRT)


# }}}
# }}}
# }}}
