import cv2
from PIL import Image
import numpy as np
from tqdm.notebook import tqdm
from pathlib import Path
import os
import sys
from moviepy.video.io.VideoFileClip import VideoFileClip


def getTotalFramesFromVideo(pathToVideo:str) -> int:
    return int(cv2.VideoCapture(pathToVideo).get(cv2.CAP_PROP_FRAME_COUNT))

def getFramesFromVideo(pathToVideo:str, fps:int=60, everyXSeconds:int=4, outputFolder:str=None, outputAllParsedFrames:bool=False, startFrame:int=None, endFrame:int=None) -> tuple:

    if outputAllParsedFrames and not outputFolder:
        raise ValueError(f"outputFolder must be specified if outputAllParsedFrames=True!")
    vidcap = cv2.VideoCapture(pathToVideo)
    if startFrame is None:  # if start isn't specified lets assume 0
        startFrame = 0
    if endFrame is None:  # if end isn't specified assume the end of the video
        endFrame = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    vidcap.set(1, startFrame)  # set the starting frame of the capture
    count = startFrame  # keep track of which frame we are up to, starting from start

    if outputFolder:
        outputDir = os.path.join(outputFolder,'allParsedFrames')
        Path(outputDir).mkdir(parents=True, exist_ok=True)

    retVal = []
    times = []
    if outputAllParsedFrames:
        print(f'INFO: outputAllParsedFrames={outputAllParsedFrames} this means all frames will be saved in {outputFolder}. This may take a while...disable this option to speed up processing.')
    for count in tqdm(range(startFrame, endFrame, int(fps*everyXSeconds))):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, count)
        if not vidcap.isOpened():
            vidcap.release()
            break
        ret, frame = vidcap.read()
        if ret:
            if outputAllParsedFrames:
                cv2.imwrite(os.path.join(outputDir, f'frame{count}.jpg'), frame)
            retVal.append(frame)
            times.append(vidcap.get(cv2.CAP_PROP_POS_MSEC))           
        else:
            vidcap.release()
            break
    
    return retVal, times

def getImage(filePath:str):
    return np.array(Image.open(filePath))



def extractClips(video:VideoFileClip, clipStartAndEndTimes:list):
    # taken from : https://stackoverflow.com/questions/75319330/how-to-extract-multiple-short-video-clips-from-a-long-video-using-python
    clip_list = []
    for (startTime, endtime) in clipStartAndEndTimes:
        clip = video.subclip(startTime, endtime)
        clip_list.append(clip)
    return clip_list

def saveVideo(video:VideoFileClip, outputFilePath:str, codec:str, threads:int, bitrate:str):
    video.write_videofile(
            outputFilePath,
            threads=threads,
            bitrate=bitrate,
            codec=codec
    )