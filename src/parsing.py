from dataclasses import dataclass, field
from datetime import timedelta
from dataclasses_json import dataclass_json
from tqdm import tqdm_notebook as tqdm
from pathlib import Path
import glob
import os
import re
from src.helpers import Coordinates, GamePhaseDetail, GamePhaseDetails, KeyItemDetail, ParsingConfig, GameSettings
from src.utils.imageUtils import getFramesFromVideo, getImage
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display
import cv2
import src.constants as constants
import numpy as np
import imagehash
import tesserocr

class ImageStore(object):
    '''
        Useful class for loading a folder of images representing different moments and key phases, then
        looking up these images later.
    '''
    
    def __init__(self, folderPath:str, gamePhaseDetails:list[GamePhaseDetail], debug:bool=False):
        '''
            :param folderPath: the base path to the folder which has a folder for each moment and a folder under each moment for each key phase's images.
                (1) FileNames MUST be delimited by "_" with the first split of the value being the string value to be returned if that image is a match.
                (2) Images MUST be stored with a jpg file extension.
                (3) moment and keyPhase names must align with BOUNDING BOXES. 
                Example: {folderPath}/{gamePhase}/{keyItem}/{side}/VALUE_Example1.jpg
        '''
        self.hashes, self.values = self._getStoredImages(folderPath, gamePhaseDetails, debug=debug)

    def lookupImageValue(self, im:Image, gamePhase:constants.GamePhase=None, keyItem:constants.KeyItem=None, side:constants.GameSide=None, maximumDistance:int=0, debug:bool=False) -> tuple:
        '''
            Looks up the image with the minimum distance (complying with minimum distance <= maximumDistance) to the provided image.
            :param im: the image to look for in the store.
            :param mom
        '''
        if len(self.hashes) == 0:
            print('WARNING: No images in the cache')
            return None, None
        imHash = self._getHash(im)
        binarydiff = self.hashes != imHash.hash.reshape((1,-1)) # calculate the hamming distance in parallel across all known hashes
        distances = binarydiff.sum(axis=1)
        minIdx = np.argmin(distances)
        minVals = np.unique(self.values[minIdx]) # get the values with the minimum hamming distance.
        if debug:
            display(f'Values:\t\t{self.values}\nDistances:\t\t{distances}')
        if len(minVals) > 1:
            raise ValueError(f'Found multiple values of equal distance: {minVals}')
        if distances[minIdx] > maximumDistance:
            return None, None
        return minVals[0], distances[minIdx]

    def _getHash(self, im:Image) :
        return imagehash.average_hash(im, hash_size=16)

    def _getStoredImages(self, folderPath:str, gamePhaseDetails:GamePhaseDetails, debug:bool=False) -> tuple :
        imgHashToValue = {}
        FILE_SEPARATOR = '_'
        for gamePhase in gamePhaseDetails.getGamePhases():
            gamePhaseFolderPath = os.path.join(folderPath, gamePhase)
            if not os.path.exists(gamePhaseFolderPath):
                continue
            for keyItemDetail in gamePhaseDetails.getGamePhaseDetail(gamePhase).getKeyItemDetails():
                keyItemFolderPath = ImageStore.getFolderPath(gamePhaseFolderPath, keyItemDetail.keyItem, keyItemDetail.side)
                if not os.path.exists(keyItemFolderPath):
                    if debug:
                        print(f'Skipping reading from `{keyItemFolderPath}`...')
                    continue
                for filename in glob.glob(os.path.join(keyItemFolderPath,'*.jpg')):
                    im = Image.open(filename)
                    im = imageTransformation(im, keyItemDetail.keyItem)
                    imhash = self._getHash(im)
                    valueRepresented = os.path.basename(filename).split(FILE_SEPARATOR)[0]
                    if debug:
                        display(f'Loaded following image with the hash=`{imhash}` and value =`{valueRepresented}` ')
                        display(im)
                    imgHashToValue[imhash] = valueRepresented
        return np.array([key.hash.flatten() for key in imgHashToValue.keys()]), np.array(list(imgHashToValue.values()))
    
    @staticmethod
    def getFolderPath(gamePhase:constants.GamePhase, keyItem:constants.KeyItem, side:constants.GameSide):
        return os.path.join(gamePhase, keyItem, side)

def getFramesFromFileContent(videoOrImageToParsePath          : str,
                             fpsOfInputVideo                  : int,
                             fileType                         : constants.FileType,
                             processEveryXSecondsFromVideo    : int,
                             outputFolder                     : str=None):
    if fileType == constants.FileType.VIDEO:
        return getFramesFromVideo(
                    pathToVideo   = videoOrImageToParsePath,
                    fps           = fpsOfInputVideo,
                    everyXSeconds = processEveryXSecondsFromVideo,
                    outputFolder  = outputFolder
                )
    elif fileType == constants.FileType.IMAGE:
        return [getImage(videoOrImageToParsePath)], [0]
    else:
        raise ValueError('Do not know how to open and process this file type.')
    return None

def getFramesFromFileContentUsingParsingConfig(parseConfig:ParsingConfig):
    return getFramesFromFileContent(
                videoOrImageToParsePath         = parseConfig.videoOrImageToParsePath,
                fpsOfInputVideo                 = parseConfig.fpsOfInputVideo,
                fileType                        = parseConfig.FileType,
                processEveryXSecondsFromVideo   = parseConfig.processEveryXSecondsFromVideo,
                outputFolder                    = parseConfig.outputFolder
            )

def drawBoxes(im:Image, keyItems:list[KeyItemDetail], outlineColor:str='blue') -> Image:
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("arial")
    for keyItem in keyItems:
        draw.rectangle(keyItem.coords.box(), outline=outlineColor)
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

    def parseTimeDuration(self, timeStr:str) -> timedelta:
        minutes, seconds = None, None
        if ':' in timeStr:
            minutes,seconds = int(timeStr.split(':')[0]), int(timeStr.split(':')[1])
        elif len(timeStr) == 4:
            try:
                minutes = int(timeStr[0])*10 + int(timeStr[1])
                seconds = int(timeStr[2])*10 + int(timeStr[3])
            except:
                raise ValueError(f'`{timeStr}` could not be parsed into time duration.')
        else:
            raise ValueError(f'`{timeStr}` could not be parsed into time duration.')
        if (seconds // 10) % 10 == 9: # OCR often reads block 2's in the time as 9's so if the ten's place is a 9 just replace it with a 2...
            seconds = 20 + (seconds % 10)
        return timedelta(minutes=int(minutes), seconds=int(seconds))

    def parseToActualValue(self, gamePhase:constants.GamePhase, keyItem:constants.KeyItem, gameSettings:GameSettings):
        if keyItem == constants.KeyItem.TIME:
            time = self.parseTimeDuration(self.parsedValue)
            if gamePhase in {constants.GamePhase.GOAL_SCORED_LEFT_HAND_SIDE, constants.GamePhase.GOAL_SCORED_RIGHT_HAND_SIDE}:
                return gameSettings.time - time - timedelta(seconds=1) # these game phases report time as how much time has passed, add 1 second to deal with rounding.
            return time #otherwise time is passed as time remaning.
        elif keyItem == constants.KeyItem.SCORE: 
            if self.parsedValue == '' or int(self.parsedValue) is None:
                raise ValueError(f'Cannot parse int from the provided value `{self.parsedValue}` for the key item {keyItem}')
            return int(self.parsedValue)
        return self.parsedValue
    
class ParsedImage:
    def __init__(self, gamePhase:constants.GamePhase, im:Image, time:float, gameSettings:GameSettings):
        self.gamePhase = gamePhase
        self.gameSettings=gameSettings
        self.image = im
        self.time = time
        self.parsingResults = {}
    
    def getKeyItemsAndSides(self):
        res = []
        for keyItem, sideValue in self.parsingResults.items():
            for side, value in sideValue.items():
                res.append((keyItem, side))
        return res
    
    def add(self, keyItem:constants.KeyItem, side:constants.GameSide, parsingResult:ParsingResult):
        if keyItem not in self.parsingResults:
            self.parsingResults[keyItem] = {} 
        self.parsingResults[keyItem][side] = parsingResult
    
    def get(self, keyItem:constants.KeyItem, side:constants.GameSide) -> ParsingResult:
        if keyItem in self.parsingResults and side in self.parsingResults[keyItem]:
            return self.parsingResults[keyItem][side]
        return None
    
    def getValue(self, keyItem:constants.KeyItem, side:constants.GameSide):
        result = self.get(keyItem, side)
        if result is None:
            return result
        try:
            return result.parseToActualValue(self.gamePhase, keyItem, self.gameSettings)
        except Exception as err:
            display(result.image)
            return None

@dataclass_json
@dataclass  
class ParsedGame:
    gameName:str
    keyMoments:dict
    parsedFrames:list[ParsedImage]

    BUFFER_IN_MILLISECONDS:int = 10000

    def getStartTime(self) -> int:
        if self.parsedFrames:
            return self.parsedFrames[0].time
        else:
            return None

    def getEndTime(self) -> int:
        if self.parsedFrames:
            return self.parsedFrames[-1].time + ParsedGame.END_GAME_BUFFER_IN_MILLISECONDS
        else:
            return None
    


class Timeline:
    '''
        Represents the timeline of a video or an individual image.
    '''
    def __init__(self, parsedFrames:list[ParsedImage], parseConfig:ParsingConfig):
        self.parseConfig=parseConfig
        self._initialize(parsedFrames)

    def _getGameNumber(self, idx:int, keyTimes:dict) -> str:
        if idx == 0 and constants.GamePhase.IN_GAME not in keyTimes:
            return 'PRE-GAME'
        elif constants.GamePhase.IN_GAME not in keyTimes:
            return f'POST-Game #{idx}'
        else:
            return f'Game #{idx}'

    def _initialize(self, parsedFrames):
        self.games = {}
        keyTimes = dict()
        frames = []
        gameNo = 0
        
        # need a buffer because inverted times can be a little off!
        BUFFER = timedelta(seconds=60)
        GOLDEN_GOAL_TIME = timedelta(seconds=120)
        lastKnownTime = None
        for idx, frame in enumerate(parsedFrames):
            if not frame or frame.gamePhase == constants.GamePhase.UNKNOWN:
                continue
            currentTime = frame.getValue(constants.KeyItem.TIME, constants.GameSide.NONE)
            if (lastKnownTime is None and currentTime is not None)\
                or (lastKnownTime is not None and currentTime is not None and (lastKnownTime + BUFFER < currentTime) and not(currentTime >= GOLDEN_GOAL_TIME - BUFFER and not(currentTime >= self.parseConfig.gameSettings.time - BUFFER))):              
                gameKey = self._getGameNumber(gameNo, keyTimes)
                self.games[gameKey] = ParsedGame(str(gameNo), keyMoments=keyTimes, parsedFrames=frames)
                gameNo += 1
                frames = []
                keyTimes = dict()
            frames.append(frame)
            if frame.gamePhase not in keyTimes:
                keyTimes[frame.gamePhase] = []
            # avoid duplicates by not having events happen at the same exact time. 
            # ASSUMPTION - cannot score 2 goals in < 1 second.
            if currentTime:
                lastKnownTime = currentTime            
            if len(keyTimes[frame.gamePhase]) > 0:
                lastFrame = keyTimes[frame.gamePhase][-1]
                lastTime = lastFrame.getValue(constants.KeyItem.TIME, constants.GameSide.NONE)
                currentTime = frame.getValue(constants.KeyItem.TIME, constants.GameSide.NONE)
                if lastTime == currentTime:
                    continue
            keyTimes[frame.gamePhase].append(frame)
        if len(frames) != 0:
            gameKey = self._getGameNumber(gameNo, keyTimes)
            self.games[gameKey] = ParsedGame(str(gameNo), keyMoments=keyTimes, parsedFrames=frames)


    def isValid(self) -> bool:
        for gameNo,game in self.games.items():
            keyTimes = game.keyMoments
            prevscoreLeft = 0
            prevscoreRight = 0
            prevTime = None
            inGame = keyTimes[constants.GamePhase.IN_GAME]
            isValid = True
            for i in range(len(inGame)):
                frame = inGame[i]
                previousFrame = inGame[i-1] if i > 0 else None
                scoreLeft = frame.getValue(constants.KeyItem.SCORE, constants.GameSide.LEFT)
                scoreRight = frame.getValue(constants.KeyItem.SCORE, constants.GameSide.RIGHT)
                time = self.parseConfig.gameSettings.time.total_seconds()-frame.getValue(constants.KeyItem.TIME, constants.GameSide.NONE).total_seconds()
                if scoreLeft is None or scoreRight is None:
                    continue
                if scoreLeft < prevscoreLeft or scoreRight < prevscoreRight:
                    isValid = False
                    display(f'Invalid frames found in {game}')
                    display(f'time =`{time}`, scoreLeft=`{scoreLeft}, scoreRight=`{scoreRight}`')
                    display(f'prevTime =`{prevTime}`, prevscoreLeft=`{prevscoreLeft}, prevscoreRight=`{prevscoreRight}`')
                    scoreLeftBox = self.parseConfig.gamePhaseDetails.getKeyItemDetail(constants.GamePhase.IN_GAME, constants.KeyItem.SCORE, constants.GameSide.LEFT).coords.box()
                    scoreRightBox = self.parseConfig.gamePhaseDetails.getKeyItemDetail(constants.GamePhase.IN_GAME, constants.KeyItem.SCORE, constants.GameSide.RIGHT).coords.box()
                    if i > 0:
                        display(game.parsedFrames[inGame[i-1]].image)
                        scoreLeftIm = previousFrame.image.crop(scoreLeftBox)
                        display(scoreLeftIm)  
                        scoreRightIm = previousFrame.image.crop(scoreRightBox)
                        display(scoreRightIm)
                    display(frame.image)
                    display(frame.image.crop(scoreLeftBox))  
                    display(frame.image.crop(scoreRightBox))  
                prevscoreLeft = scoreLeft
                prevscoreRight = scoreRight
                prevTime = time
        return isValid

    def _evaluateGame(self, game:ParsedGame):
        result = {}
        for gamePhase,frames in game.keyMoments.items():
            totalFrames = 0
            missedFrames = 0
            if gamePhase not in {constants.GamePhase.IN_GAME, constants.GamePhase.IN_GAME_FINAL_RESULT_LEFT_HAND_SIDE,constants.GamePhase.IN_GAME_FINAL_RESULT_RIGHT_HAND_SIDE, constants.GamePhase.GOAL_SCORED_LEFT_HAND_SIDE, constants.GamePhase.GOAL_SCORED_RIGHT_HAND_SIDE}:
                result[gamePhase] = 100
                continue
            for i in range(1,len(frames)):
                totalFrames += 1
                currentScoreLeft = frames[i].getValue(constants.KeyItem.SCORE, constants.GameSide.LEFT) 
                oldScoreLeft = frames[i-1].getValue(constants.KeyItem.SCORE, constants.GameSide.LEFT)
                currentScoreRight = frames[i].getValue(constants.KeyItem.SCORE, constants.GameSide.RIGHT) 
                oldScoreRight = frames[i-1].getValue(constants.KeyItem.SCORE, constants.GameSide.RIGHT)
                if (currentScoreLeft is not None and oldScoreLeft is not None and currentScoreLeft - oldScoreLeft   > 1)\
                or (currentScoreRight is not None and oldScoreRight is not None and currentScoreRight - oldScoreRight   > 1):
                    missedFrames += 1
                    totalFrames += 1
            if missedFrames > 0 and totalFrames > 0:
                result[gamePhase] = missedFrames*100.0/totalFrames 
            elif missedFrames==0 and totalFrames ==0:
                result[gamePhase] = 100
            else:
                raise ValueError(f'missedFrames > 0 but totalFrames is 0.')
        return result


    def evaluateQuality(self):
        quality = {}
        for game in self.keyTimes.keys():
            quality[game] = self._evaluateGame(self.keyTimes[game])
        display(quality)

    def dumpAllUnparseableImages(self, folderPath:str):
        cnt = 0
        for frame in self.parsedFrames:
            keyItemsAndSides = frame.getKeyItemsAndSides()
            for keyItem, side in keyItemsAndSides:
                subFolderPath = ImageStore.getFolderPath(frame.gamePhase, keyItem, side)
                subFolderCompletePath = os.path.join(folderPath, subFolderPath)
                im = frame.get(keyItem, side).image
                parsedActualValue = frame.getValue(keyItem, side)
                if parsedActualValue is None:
                    cnt += 1
                    Path(subFolderCompletePath).mkdir(parents=True, exist_ok=True)
                    print(f'storing in {subFolderCompletePath}')
                    im.save(os.path.join(subFolderCompletePath, f'UNKNOWN_Example_{cnt}.jpg'))


def runOcrOnImage(im:Image, gamePhase:constants.GamePhase, keyItemDetail:KeyItemDetail, debug:bool=False, storedImageCache:ImageStore=None) -> str:
    if storedImageCache:
        value, dist = storedImageCache.lookupImageValue(im, gamePhase, keyItemDetail.keyItem, keyItemDetail.side, keyItemDetail.maximumDistanceForStoredImages)
        if value is not None:
            if debug:
                print(f'Found item in STORED IMAGES with distance of `{dist}` and value of `{value}`')
            return value, [dist]
    with tesserocr.PyTessBaseAPI() as api:
        api.SetPageSegMode(tesserocr.PSM.SINGLE_LINE)
        for option,value in keyItemDetail.tesserocrOptions.items():
            api.SetVariable(option, value)
        if keyItemDetail.numbersOnly:
            api.SetVariable("tessedit_char_whitelist", "0123456789")
        api.SetImage(im)
        res = api.GetUTF8Text()
        return re.sub(r'\n', '', res), api.AllWordConfidences()


def tryParseImage(im:Image, time:float, gamePhaseDetail:GamePhaseDetail, gameSettings:GameSettings, debug:bool=False, ignoreErrors:bool=False,  storedImageCache:ImageStore=None, displayImages:bool=False) -> ParsedImage:
    parsedImage = ParsedImage(gamePhaseDetail.gamePhase, im, time, gameSettings=gameSettings)
    if debug:
        display(f"[{gamePhaseDetail.gamePhase}]:  Extracting the following areas from the image:")
        #display(drawBoxes(im.copy(), gamePhaseDetail.keyItemDetails))
    for keyItem in gamePhaseDetail.getKeyItemDetails():
        box = keyItem.coords.box()
        cropped = im.crop(box)
        if debug:
            print(f'[{gamePhaseDetail.gamePhase}]: Outputting cropped image for {keyItem.keyItem} side {keyItem.side}, box={box}')
            if displayImages:
                display(cropped)
        transformed = None
        try:
            transformed = imageTransformation(cropped, keyItem.keyItem)
        except Exception as err:
            print(f'[{gamePhaseDetail.gamePhase}]:  Exception encountered when trying to transform image for {keyItem.keyItem} side {keyItem.side}')
            raise err
        parsedVal, confidences = runOcrOnImage(transformed, gamePhaseDetail.gamePhase, keyItem, storedImageCache=storedImageCache, debug=debug)
        parsingResult = ParsingResult(image=cropped, parsedValue=parsedVal, confidences=confidences)
        if debug:
            display(f'[{gamePhaseDetail.gamePhase}]:  ParsedValue: `{parsedVal}`')
            display(f'[{gamePhaseDetail.gamePhase}]:  Confidences: {confidences}')
            if displayImages:
                display(f'[{gamePhaseDetail.gamePhase}]:  Image parsed:')
                display(transformed)
        try:
            parsingResult.parseToActualValue(gamePhase=gamePhaseDetail.gamePhase, keyItem=keyItem.keyItem, gameSettings=gameSettings)
        except Exception as err:
            if debug:
                print(f'[{gamePhaseDetail.gamePhase}]: Exception when parsing {keyItem.keyItem}. Text=`{parsedVal}`')
            if ignoreErrors:
                parsedImage.add(keyItem=keyItem, parsingResult=parsingResult)
                continue        
            if gamePhaseDetail.identifyingKeyItem[0] == keyItem.keyItem and gamePhaseDetail.identifyingKeyItem[1] == keyItem.side:
                print(f'[{gamePhaseDetail.gamePhase}]:  Was not {gamePhaseDetail.gamePhase} because {keyItem.keyItem} could not be parsed. Text=`{parsedVal}`')
                print(f'[{gamePhaseDetail.gamePhase}]:  {err}')
                return None, False
        parsedImage.add(keyItem=keyItem.keyItem, side=keyItem.side, parsingResult=parsingResult)
    return parsedImage, True

def tryProcessFrame(frame:np.array, time:float, gamePhaseDetails:GamePhaseDetails, gameSettings:GameSettings, debug:bool=False, ignoreErrors:bool=False, storedImageCache:ImageStore=None, displayImages:bool=False) -> tuple:
    im = Image.fromarray(frame)
    result = None
    for gamePhaseDetail in gamePhaseDetails.getGamePhaseDetails():
        result, success = tryParseImage(im, time, gamePhaseDetail, gameSettings, debug, ignoreErrors, storedImageCache=storedImageCache, displayImages=displayImages)
        if success:
            return result, True
    return ParsedImage(constants.GamePhase.UNKNOWN, im, time, gameSettings), False

def process(frames:list[np.array], times:list, gamePhaseDetails:GamePhaseDetails, gameSettings:GameSettings,debug:bool=False, ignoreErrors:bool=False, storedImageCache:ImageStore=None) -> list[ParsedImage]:
    results = []
    successes = 0
    print(f'Starting to process frames...')
    for frame, time in tqdm(zip(frames, times), total=len(frames)):
        result, success = tryProcessFrame(frame, time, gamePhaseDetails, gameSettings, debug, ignoreErrors, storedImageCache=storedImageCache)
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
    frames, times = getFramesFromFileContentUsingParsingConfig(parseConfig)
    print(f'Successfully got frames from the file `{parseConfig.videoOrImageToParsePath}`')
    storedImageCache = None
    if parseConfig.imageCacheFolderPath is not None:
        storedImageCache = ImageStore(parseConfig.imageCacheFolderPath, parseConfig.gamePhaseDetails)
    results = process(frames, times, parseConfig.gamePhaseDetails, parseConfig.gameSettings, parseConfig.debug, parseConfig.ignoreParsingErrors, storedImageCache=storedImageCache)
    timeline = Timeline(parsedFrames=results, parseConfig=parseConfig)
    if parseConfig.debug:
        return timeline, results
    return timeline, None
    