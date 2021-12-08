#!/usr/bin/python3
import cv2 as cv  # NOQA
import numpy as np  # NOQA
import matplotlib.pyplot as plt  # NOQA
import matplotlib  # NOQA
from scipy.optimize import curve_fit  # NOQA
from scipy.signal import find_peaks  # NOQA
import os

plt.switch_backend('TkAgg')

def readVideo(filePath, **kwargs):
    # {{{
    roi = kwargs.get("roi", None)

    displayVideo = kwargs.get("displayVideo", True)
    rescaleFactor = kwargs.get("rescale", 1.0)
    windowName = kwargs.get("windowName", filePath)
    waitKeyTime = kwargs.get("waitKeyTime", 1)

    avgIntenList = []
    timeMillisList = []

    cap = cv.VideoCapture(filePath)
    while True:
        status, frame = cap.read()
        if status:
            if displayVideo:
                frame = rescaleFrame(frame, rescaleFactor)
                frame = drawRoi(frame, roi)

                cv.imshow(windowName, frame)

                key = cv.waitKey(waitKeyTime)

                if key == 32:
                    roi = cv.selectROI(windowName, frame)
                    print(roi)
                elif key == ord("q"):
                    break

            if roi is not None:
                if roi == "all":
                    channelsAvgInten = cv.mean(frame)[:3]
                else:
                    channelsAvgInten = calcAvgInten(frame, roi)
                timeMillis = cap.get(cv.CAP_PROP_POS_MSEC)
                avgIntenList.append(channelsAvgInten)
                timeMillisList.append(timeMillis)

        else:
            break

    cap.release()
    cv.destroyAllWindows()

    if avgIntenList == []:
        raise RuntimeError(
            "Array of average intensities is empty! Did you select or pass"
            "as an argument a region of interest (roi)?"
        )

    avgIntenArr = np.array(avgIntenList)
    timeScdsArr = np.array(timeMillisList) / 1000

    if kwargs.get("plotAllChannels", False):
        plotAvgIntens(timeScdsArr, avgIntenArr, **kwargs)

    return timeScdsArr, avgIntenArr


# }}}


def readCamera(cameraNum, **kwargs):    # NOQA
# {{{
    cap = cv.VideoCapture(cameraNum)

    cameraRes = kwargs.get("cameraRes", (cap.get(3), cap.get(4)))
    windowName = kwargs.get("windowName", f"Cam {cameraNum} at {cameraRes}")
    waitKeyTime = kwargs.get("waitKeyTime", 1)
    recordVideo = kwargs.get("recordVideo", False)
    filePath = kwargs.get("filePath", os.getcwd() + "/")
    fileName = kwargs.get("fileName", "")
    roi = kwargs.get("roi", None)

    if recordVideo:
        writer = cv.VideoWriter(
            filePath + fileName + 'capture.wmv',
            cv.VideoWriter_fourcc(*'DIVX'), 30, cameraRes
        )

    avgIntenList = []
    timeMillisList = []
    startTime = False

    while True:
        status, frame = cap.read()
        if status:
            if recordVideo:
                writer.write(frame)

            frame = drawRoi(frame, roi)
            cv.imshow(windowName, frame)

            key = cv.waitKey(waitKeyTime)

            if key == 32:
                roi = cv.selectROI(windowName, frame)
                print(roi)
            elif key == ord("s"):
                if not startTime:
                    kwargs['fromTime'] = cap.get(cv.CAP_PROP_POS_MSEC)/1000
                    print(f"from time: {kwargs['fromTime']}")
                    startTime = True
                else:
                    kwargs['toTime'] = cap.get(cv.CAP_PROP_POS_MSEC)/1000
                    print(f"to time: {kwargs['toTime']}")
                    startTime = False
            elif key == ord("q"):
                break

            if roi is not None:
                if roi == "all":
                    channelsAvgInten = cv.mean(frame)[:3]
                else:
                    channelsAvgInten = calcAvgInten(frame, roi)
                timeMillis = cap.get(cv.CAP_PROP_POS_MSEC)
                avgIntenList.append(channelsAvgInten)
                timeMillisList.append(timeMillis)

        else:
            break

    cap.release()
    cv.destroyAllWindows()

    if avgIntenList == []:
        raise RuntimeError(
            "Array of average intensities is empty! Did you select or pass"
            "as an argument a region of interest (roi)?"
        )

    avgIntenArr = np.array(avgIntenList)
    timeScdsArr = np.array(timeMillisList) / 1000

    if kwargs.get("plotAllChannels", False):
        plotAvgIntens(timeScdsArr, avgIntenArr, **kwargs)

    return timeScdsArr, avgIntenArr
# }}}


def fitFuncs(timeScdsArr, avgIntenArr, **kwargs):
    # {{{
    channelToUse = kwargs.get("channelToUse", "g")

    B, G, R = separateChannels(avgIntenArr)
    channelsDict = {"b": B, "g": G, "r": R}
    channelAvgIntenArr = channelsDict[channelToUse]

    timeScdsArr, channelAvgIntenArr = shiftArr(
        timeScdsArr, channelAvgIntenArr, **kwargs
    )

    expGuesses = kwargs.get("expGuesses", ["maxInten", -0.5, 0])
    if isinstance(expGuesses[0], str):
        expGuesses[0] = channelAvgIntenArr[0]
    expParams, expStdDev = fitExponential(
        timeScdsArr, channelAvgIntenArr, p0=expGuesses
    )

    polyGuesses = kwargs.get("polyGuesses", [0, 0, 0, 0, 0, 0])
    polyParams, polyStdDev = fitPolynomial(
        timeScdsArr, channelAvgIntenArr, p0=polyGuesses
    )

    maxDivergenceIndexList = findMaxDivergencePeaks(timeScdsArr, expParams, polyParams)

    rcrtGuesses = kwargs.get("rcrtGuesses", expParams)
    rcrtParams, rcrtStdDev = None, None
    for i in maxDivergenceIndexList:
        try:
            rcrtParams, rcrtStdDev = fitRCRT(
                timeScdsArr, channelAvgIntenArr, i, p0=rcrtGuesses
            )
            maxDivergenceIndex = i
            break
        except RuntimeError:
            continue

    if rcrtParams is None or rcrtStdDev is None:
        raise RuntimeError(f"rcrt fit failed with p0={rcrtGuesses}")

    funcParamsDict = {
        "rCRT": (rcrtParams, rcrtStdDev),
        "exp": (expParams, expStdDev),
        "poly": (polyParams, polyStdDev),
        "maxDiv": maxDivergenceIndex,
        "timeScdsArr": timeScdsArr,
        "avgIntenArr": channelAvgIntenArr,
    }

    return funcParamsDict


# }}}


def calcRCRT(arg, **kwargs):
    # {{{
    try:
        timeScdsArr, avgIntenArr = readCamera(int(arg), **kwargs)
    except ValueError:
        timeScdsArr, avgIntenArr = readVideo(arg, **kwargs)
    funcParamsDict = fitFuncs(timeScdsArr, avgIntenArr, **kwargs)

    rcrt, rcrtUncertainty = rCRTFromParams(funcParamsDict["rCRT"])

    if kwargs.get("plotRCRT", True):
        plotRCRT(funcParamsDict, **kwargs)

    return rcrt, rcrtUncertainty


# }}}


def rCRTFromParams(rcrtParams):
# {{{
    inverseRCRT = rcrtParams[0][1]
    inverseRCRTStdDev = rcrtParams[1][1]

    rcrt = -1 / inverseRCRT
    rcrtStdDev = -4 * rcrt * (inverseRCRTStdDev/inverseRCRT)

    return (rcrt, rcrtStdDev)
# }}}


def setFigureSizePx(figSizePx):
    # {{{
    px = 1 / plt.rcParams["figure.dpi"]
    plt.rcParams["figure.figsize"] = (figSizePx[0] * px, figSizePx[1] * px)


# }}}


def plotAvgIntens(timeArr, avgIntenArr, **kwargs):
    # {{{
    setFigureSizePx(kwargs.get("figSizePx", (1280, 720)))

    for intenList, color in zip(separateChannels(avgIntenArr), "bgr"):
        plt.plot(timeArr, intenList, color=color, label="avgInten")
    plt.xlabel("Time (s)")
    plt.ylabel("Average intensity (a.u.)")
    plt.title("Channels b, g and r")

    filePath = kwargs.get("filePath", os.getcwd() + "/")
    fileName = kwargs.get("fileName", "")

    plotBoundaries(timeArr, **kwargs)
    plt.legend()

    if kwargs.get("saveFigs", False):
        plt.savefig(filePath + fileName + "all-channels.png", bbox_inches="tight")
    if kwargs.get("showPlots", True):
        plt.tight_layout()
        plt.show()
    else:
        plt.close("all")


# }}}


def plotBoundaries(timeArr, **kwargs):
# {{{
    if kwargs.get("fromTime", False):
        plt.axvline(
            kwargs["fromTime"],
            c="k",
            ls=":",
            label="start",
        )
    if kwargs.get("toTime", False):
        plt.axvline(
            kwargs["toTime"],
            c="k",
            ls=":",
            label="end",
        )
# }}}


def plotRCRT(funcParamsDict, **kwargs):
    # {{{
    setFigureSizePx(kwargs.get("figSizePx", (1280, 720)))
    channelToUse = kwargs.get("channelToUse", "g")

    timeScdsArr = funcParamsDict["timeScdsArr"]
    channelAvgIntenArr = funcParamsDict["avgIntenArr"]
    expParams, expStdDev = funcParamsDict["exp"]
    polyParams, polyStdDev = funcParamsDict["poly"]
    rcrtParams, rcrtStdDev = funcParamsDict["rCRT"]

    expY = exponential(timeScdsArr, *expParams)
    polyY = polynomial(timeScdsArr, *polyParams)
    rcrtY = exponential(timeScdsArr, *rcrtParams)
    rcrt, rcrtUncertainty = rCRTFromParams(funcParamsDict["rCRT"])

    plt.plot(timeScdsArr, channelAvgIntenArr, f"{channelToUse}-", label="avgIntens")
    plt.plot(timeScdsArr, expY, "--", label="exp")
    plt.plot(timeScdsArr, polyY, "--", label="poly")
    plt.plot(timeScdsArr, rcrtY, "c-", label="crtExp")
    plt.axvline(
        timeScdsArr[funcParamsDict["maxDiv"]],
        c="k",
        ls=":",
        label="cutoff",
    )

    plt.plot([], [], " ", label="rCRT")  # Hack

    np.set_printoptions(precision=3)
    plt.legend(
        [
            f"Channel {channelToUse}",
            f"exp params: {expParams} +- {expStdDev}",
            f"poly params: {polyParams} +- {polyStdDev}",
            f"crt params: {rcrtParams} +- {rcrtStdDev}",
            "Máxima divergência",
            f"rCRT: {round(rcrt, 2)} +- {round(rcrtUncertainty, 2)}",
        ],
        fontsize="8",
    )
    plt.xlabel("Tempo (s)")
    plt.ylabel("Intensidade média (u.a.)")
    plt.title(f"Channel {channelToUse} and functions")

    filePath = kwargs.get("filePath", os.getcwd() + "/")
    fileName = kwargs.get("fileName", "")

    if kwargs.get("saveFigs", False):
        plt.savefig(filePath + fileName + "rCRT.png", bbox_inches="tight")
    if kwargs.get("showPlots", True):
        plt.tight_layout()
        plt.show()
    else:
        plt.close("all")


# }}}


def shiftArr(timeArr, arr, **kwargs):
    # {{{
    fromTime = kwargs.get("fromTime", "channel max")
    if fromTime == "channel max":
        fromIndex = np.argmax(arr)
    elif isinstance(fromTime, int):
        fromIndex = np.where(timeArr >= fromTime)[0][0]

    toTime = kwargs.get("toTime", "end")
    if toTime == "end":
        toIndex = len(timeArr)
    elif isinstance(toTime, int):
        toIndex = np.where(timeArr <= toTime)[0][0]

    return (
        timeArr[fromIndex:toIndex] - np.amin(timeArr[fromIndex:toIndex]),
        arr[fromIndex:toIndex] - np.amin(arr[fromIndex:toIndex]),
    )


# }}}


def covToStdDev(cov):
    return 4 * np.sqrt(np.diag(cov))


def findMaxDivergencePeaks(timeScdsArr, expParams, polyParams):
    # {{{
    diffArray = diffExpPoly(timeScdsArr, expParams, polyParams)
    maxIndexes = find_peaks(diffArray)[0]
    maxIndexesSorted = sorted(maxIndexes, key=lambda x: diffArray[x], reverse=True)
    return maxIndexesSorted


# }}}


def diffExpPoly(x, expParams, polyParams):
    return np.abs(exponential(x, *expParams) - polynomial(x, *polyParams))


def exponential(x, a, b, c):
    return a * np.exp(b * x) + c


def polynomial(x, *coefs):
    return np.polynomial.Polynomial(list(coefs))(x)


def fitExponential(timeScdsArr, channelIntenArr, p0=[0, 0, 0]):
    # {{{
    try:
        expParams, expCov = curve_fit(
            f=exponential,
            xdata=timeScdsArr,
            ydata=channelIntenArr,
            p0=p0,
            bounds=(-np.inf, np.inf),
        )
        expStdDev = covToStdDev(expCov)
        return expParams, expStdDev
    except RuntimeError:
        raise RuntimeError(f"exponential fit failed with p0={p0}")


# }}}


def fitPolynomial(timeScdsArr, channelIntenArr, p0=[0, 0, 0, 0, 0, 0]):
    # {{{
    try:
        polyParams, polyCov = curve_fit(
            f=polynomial,
            xdata=timeScdsArr,
            ydata=channelIntenArr,
            p0=p0,
            bounds=(-np.inf, np.inf),
        )
        polyStdDev = covToStdDev(polyCov)
        return polyParams, polyStdDev
    except RuntimeError:
        raise RuntimeError(f"polynomial fit failed with p0={p0}")


# }}}


def fitRCRT(timeScdsArr, channelIntenArr, maxDivergenceIndex, p0=[0, 0, 0]):
    # {{{
    try:
        rcrtParams, rcrtStdDev = fitExponential(
            timeScdsArr[:maxDivergenceIndex],
            channelIntenArr[:maxDivergenceIndex],
            p0=p0,
        )
        return rcrtParams, rcrtStdDev
    except RuntimeError:
        raise RuntimeError(f"rcrt fit failed with p0={p0}")


# }}}


def separateChannels(avgIntenArr):
    # {{{
    B = avgIntenArr[:, 0]
    G = avgIntenArr[:, 1]
    R = avgIntenArr[:, 2]
    return (B, G, R)


# }}}


def calcAvgInten(frame, roi):
    # {{{
    croppedFrame = cropFrame(frame, roi)
    channelsAvgInten = cv.mean(croppedFrame)[:3]
    return channelsAvgInten


# }}}


def cropFrame(frame, roi):
    # {{{
    x1, y1, sideX, sideY = roi
    return frame[y1 : y1 + sideY, x1 : x1 + sideX]


# }}}


def drawRoi(frame, roi):
    # {{{
    if isinstance(roi, tuple):
        x1, y1, sideX, sideY = roi
        x2, y2 = x1 + sideX, y1 + sideY
        return cv.rectangle(frame, (x1, y1), (x2, y2), [0, 0, 255], 2)
    else:
        return frame
# }}}


def rescaleFrame(frame, factor):
    return cv.resize(frame, (0, 0), fx=factor, fy=factor)


def changeCameraRes(cap, width, height):
# {{{
    cap.set(3, width)
    cap.set(4, height)
# }}}


if __name__ == "__main__":
    from sys import argv

    try:
        calcRCRT(
            argv[1],
            displayVideo=True,
            rescale=1.0,
            plotAllChannels=True,
            roi="all",
            channelToUse="g",
        )
    except IndexError:
        calcRCRT(
            0,
            displayVideo=True,
            rescale=1.0,
            plotAllChannels=True,
            roi="all",
            channelToUse="g",
        )
