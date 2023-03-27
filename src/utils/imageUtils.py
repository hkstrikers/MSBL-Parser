import cv2
from PIL import Image
import numpy as np

def getFramesFromVideo(pathToVideo:str, fps:int=60, everyXSeconds:int=4, outputFolder:str=None) -> tuple:
    vidcap = cv2.VideoCapture(pathToVideo)
    success,image = vidcap.read()
    count = 0

    retVal = []
    times = []
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if ret:
            if outputFolder:
                cv2.imwrite(f'{outputFolder}/frame{count}.jpg', frame)
            retVal.append(frame)
            times.append(vidcap.get(cv2.CAP_PROP_POS_MSEC))
            count += fps*everyXSeconds # i.e. at 30 fps, this advances one second
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, count)
        else:
            vidcap.release()
            break
    
    return retVal, times

def getImage(filePath:str):
    return np.array(Image.open(filePath))

