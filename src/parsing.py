from datetime import timedelta
import re
from src.helpers import Coordinates, GamePhaseDetail, KeyItemDetail, ParsingConfig
from src.utils.imageUtils import getFramesFromVideo, getImage
from PIL import Image, ImageDraw, ImageFont
from IPython.display import Image, display
import cv2
import constants
import numpy as np
import imagehash
import tesserocr



def getFramesFromFileContent(videoOrImageToParsePath          : str,
                             fpsOfInputVideo                  : int,
                             fileType                         : constants.FileType,
                             processEveryXSecondsFromVideo    : int,
                             outputFolder                     : str=None):
    frames = []
    if fileType == constants.FileType.VIDEO:
        frames = getFramesFromVideo(
                    pathToVideo   = videoOrImageToParsePath,
                    fps           = fpsOfInputVideo,
                    everyXSeconds = processEveryXSecondsFromVideo,
                    outputFolder  = outputFolder
                )
    elif fileType == constants.FileType.IMAGE:
        return [getImage(videoOrImageToParsePath)]
    else:
        raise ValueError('Do not know how to open and process this file type.')
    return None


def drawBoxes(im:Image, keyItems:list[KeyItemDetail]) -> Image:
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("arial")
    for keyItem in keyItems:
        draw.rectangle(keyItem.box, outline='red')
    return im

def imageTransformationForScore(im:Image) -> Image:
    tmp = im
    tmp = cv2.cvtColor(np.array(tmp), cv2.COLOR_BGR2GRAY)
    (th, newimg) = cv2.threshold(tmp, 215, 255, cv2.THRESH_BINARY)
    #lower = np.array([0,0,196])
    #upper = np.array([179,39,255])
    #mask = cv2.inRange(tmp, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
    opening = cv2.morphologyEx(newimg, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(opening,kernel,iterations = 2)
    return Image.fromarray(255-erosion)

def imageTransformation(im:Image, type:constants.KeyItem) -> list:
    if type in {constants.KeyItem.SCORE}:
        return imageTransformationForScore(im)
    elif type in {constants.KeyItem.TIME}:
        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(np.array(im),kernel,iterations = 1)
        im1 = Image.fromarray(erosion)
        return im1
    elif type in {constants.KeyItem.TEAM_NAME}:
        tmp = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2GRAY)
        (th, newimg) = cv2.threshold(tmp, 50, 255, cv2.THRESH_BINARY)
        return Image.fromarray(newimg)
    else:
        print('No transformation found')
        return im

@dataclass
class ParsingResult:
    image:Image
    parsedValue:str
    confidences:list

    def parseTimeDuration(timeStr:str) -> timedelta:
        minutes, seconds = None, None
        if ':' in timeStr:
            minutes,seconds = timeStr.split(':')[0], timeStr.split(':')[1]
        elif len(timeStr) == 4:
            minutes = int(timeStr[0])*10 + int(timeStr[1])
            seconds = int(timeStr[2])*10 + int(timeStr[3])
        else:
            raise ValueError(f'`{timeStr}` could not be parsed into time duration.')
        return timedelta(minutes=int(minutes), seconds=int(seconds))

    def parseToActualValue(self, keyItem:constants.KeyItem):
        if keyItem == constants.KeyItem.TIME:
            return self.parseTimeDuration(self.parsedValue)
        elif keyItem == constants.KeyItem.SCORE: 
            if self.parsedValue == '' or int(self.parsedValue) is None:
                raise ValueError(f'Cannot parse int from the provided value `{self.parsedValue}` for the key item {keyItem}')
            return int(self.parsedValue)
        return self.parsedValue
    
class ParsedImage:
    def __init__(self, gamePhase:constants.GamePhase):
        self.gamePhase = gamePhase
        self.parsingResults = {}
    
    def add(self, keyItem:constants.KeyItem, parsingResult:ParsingResult):
        self.parsingResults[keyItem] = parsingResult
    
    def get(self, keyItem:constants.KeyItem) -> ParsingResult:
        if keyItem in self.parsingResults:
            return self.parsingResults[keyItem]
        return None

def runOcrOnImage(im:Image, keyItem:KeyItemDetail, debug:bool=False) -> str:
    value, dist = STORED_IMAGES.lookupImageValue(im, moment, item.name)
    if value is not None:
        if debug:
            print(f'Found item in STORED IMAGES with distance of `{dist}` and value of `{value}`')
        return value
    with tesserocr.PyTessBaseAPI() as api:
        api.SetPageSegMode(tesserocr.PSM.SINGLE_LINE)
        for option,value in keyItem.tesserocrOptions.items():
            api.SetVariable(option, value)
        if keyItem.numbersOnly:
            api.SetVariable("tessedit_char_whitelist", "0123456789")
        api.SetImage(im)
        res = api.GetUTF8Text()
        return re.sub(r'\n', '', res), api.AllWordConfidences()


def tryParseImage(im:Image, gamePhaseDetail:GamePhaseDetail, debug:bool=False) -> ParsedImage:
    parsedImage = ParsedImage(gamePhaseDetail)
    if debug:
        display("Extracting the following areas from the image:")
        display(drawBoxes(im.copy(), gamePhaseDetail.keyItemDetails))
    for keyItem in gamePhaseDetail.keyItemDetails:
        box = keyItem.coords.box()
        cropped = im.crop(box)
        if debug:
            print(f'Outputting cropped image for {keyItem.keyItem} side {keyItem.side}, box={box}')
            display(cropped)
        transformed = cropped
        try:
            transformed = imageTransformation(cropped, keyItem.keyItem)
        except Exception as err:
            print(f'Exception encountered when trying to transform image for {keyItem.keyItem} side {keyItem.side}')
            raise err
        parsedVal, confidences = runOcrOnImage(transformed, keyItem.numbersOnly)
        parsingResult = ParsingResult(image=transformed, parsedValue=parsedVal, confidences=confidences)
        if debug:
            display(f'ParsedValue: `{parsedVal}`')
            display(f'Confidences: {confidences}')
            display('Image parsed:')
            display(transformed)
        try:
            parsingResult.parseToActualValue(keyItem=keyItem.keyItem)
        except Exception as err:
            if debug:
                print(f'Exception when parsing {keyItem.keyItem}. Text=`{parsedVal}`')
            if gamePhaseDetail.identifyingKeyItem == keyItem.keyItem:
                print(f'Was not {gamePhaseDetail.gamePhase} because {keyItem.keyItem} could not be parsed. Text=`{parsedVal}`')
                print(f'{err}')
                return None, False
        parsedImage.add(keyItem=keyItem, parsingResult=parsingResult)
    return parsedImage, True

def tryProcessFrame(frame:np.array, gamePhaseDetails:list[GamePhaseDetail], debug:bool=False) -> tuple:
    im = Image.fromarray(frame)
    result = None
    for gamePhaseDetail in gamePhaseDetails:
        result, success = tryParseImage(frame, gamePhaseDetail, debug)
        if success:
            return result, True
    return None, False

def process(frames:list[np.array], gamePhaseDetails:list[GamePhaseDetail], debug:bool=False) -> list[ParsedImage]:
    results = []
    successes = 0
    for frame in frames:
        result, success = tryProcessFrame(frame, gamePhaseDetails, debug)
        if not success:
            if debug:
                display(f'Unable to process the following frame, not recognized as any provided game phase:')
                display(frame)
        else:
            successes += 1
        results.append(result)
    print(f'Successfully processed {successes*100.0/len(frames)}% ({successes} out of {len(frames)}) frames.')
    return results

    

def parse(parseConfig:ParsingConfig):
    frames = getFramesFromFileContent(
                videoOrImageToParsePath         = parseConfig.videoOrImageToParsePath,
                fpsOfInputVideo                 = parseConfig.fpsOfInputVideo,
                fileType                        = parseConfig.FileType,
                processEveryXSecondsFromVideo   = parseConfig.processEveryXSecondsFromVideo,
                outputFolder                    = parseConfig.outputFolder
            )
    results = process(frames, parseConfig.gamePhaseDetails, parseConfig.debug)
    return results
    