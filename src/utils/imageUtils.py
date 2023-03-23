import cv2

def getFramesFromVideo(pathToVideo:str, fps:int=60, everyXSeconds:int=4, outputPath:str='', shouldOutputToFile:bool=False) -> list:
    vidcap = cv2.VideoCapture(pathToVideo)
    success,image = vidcap.read()
    count = 0

    retVal = []
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if ret:
            if shouldOutputToFile:
                cv2.imwrite(f'{outputPath}/frame{count}.jpg', frame)
            retVal.append(frame)
            count += fps*everyXSeconds # i.e. at 30 fps, this advances one second
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, count)
        else:
            vidcap.release()
            break
    
    return retVal