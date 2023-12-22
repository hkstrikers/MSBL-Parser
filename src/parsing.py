from dataclasses import dataclass, field
from datetime import timedelta
import json
from dataclasses_json import dataclass_json
from tqdm.notebook import tqdm
from pathlib import Path
import glob
import os
import re
from config import PARSE_CONFIG
from helpers import Coordinates, GamePhaseDetail, GamePhaseDetails, KeyItemDetail, ParsingConfig, GameSettings, DEFAULT_GAME_PHASE_DETAILS
from utils.imageUtils import getFramesFromVideo, getImage, extractClips, saveVideo, getTotalFramesFromVideo
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2
import constants as constants
import numpy as np
import imagehash
import tesserocr
import multiprocessing
from typing_extensions import Self
import sys
TESSERDATA_PATH = '/usr/share/tesseract-ocr/4.00/tessdata' #os.path.join(sys.executable, "..", "share", "tessdata")

class ImageStore(object):
    '''
        Useful class for loading a folder of images representing different moments and key phases, then
        looking up these images later.
    '''
    
    def __init__(self, folderPath:str, gamePhaseDetails:GamePhaseDetails, debug:bool=False):
        '''
            :param folderPath: the base path to the folder which has a folder for each moment and a folder under each moment for each key phase's images.
                (1) FileNames MUST be delimited by "_" with the first split of the value being the string value to be returned if that image is a match.
                (2) Images MUST be stored with a jpg file extension.
                (3) moment and keyPhase names must align with BOUNDING BOXES. 
                Example: {folderPath}/{gamePhase}/{keyItem}/{side}/VALUE_Example1.jpg
        '''
        self.hashes, self.values = self._getStoredImages(folderPath, gamePhaseDetails, debug=debug)

    def lookupImageValue(self, im:Image, maximumDistance:int=0, debug:bool=False) -> tuple:
        '''
            Looks up the image with the minimum distance (complying with minimum distance <= maximumDistance) to the provided image.
            :param im: the image to look for in the store.
            :param mom
        '''
        if len(self.hashes) == 0:
            if debug:
                print('[Image cache] WARNING: No images in the cache')
            return None, None
        imHash = self._getHash(im)
        binarydiff = self.hashes != imHash.hash.reshape((1,-1)) # calculate the hamming distance in parallel across all known hashes
        distances = binarydiff.sum(axis=1)
        minIdx = np.argmin(distances)
        minVals = np.unique(self.values[minIdx]) # get the values with the minimum hamming distance.
        if debug:
            display(f'[Image cache]  Values:\t\t{self.values}\nDistances:\t\t{distances}')
        if len(minVals) > 1:
            raise ValueError(f'[Image cache] Found multiple values of equal distance: {minVals}')
        if distances[minIdx] > maximumDistance:
            return None, None
        return minVals[0].item(), distances[minIdx].item()

    def _getHash(self, im:Image) :
        return imagehash.average_hash(im, hash_size=16)

    def _getStoredImages(self, folderPath:str, gamePhaseDetails:GamePhaseDetails, debug:bool=False) -> tuple :
        imgHashToValue = {}
        FILE_SEPARATOR = '_'
        for gamePhase in gamePhaseDetails.getGamePhases():
            gamePhaseFolderPath = os.path.join(folderPath, gamePhase)
            if not os.path.exists(gamePhaseFolderPath):
                if debug:
                    print(f'Skipping reading from `{gamePhaseFolderPath}`...')
                continue
            for keyItemDetail in gamePhaseDetails.getGamePhaseDetail(gamePhase).getKeyItemDetails():
                keyItemFolderPath = ImageStore.getFolderPath(gamePhaseFolderPath, keyItemDetail.keyItem, keyItemDetail.side)
                if debug:
                    print(f'Trying to read from {keyItemFolderPath}')
                if not os.path.exists(keyItemFolderPath):
                    if debug:
                        print(f'Skipping reading from `{keyItemFolderPath}`...')
                    continue
                for filename in glob.glob(os.path.join(keyItemFolderPath,'*.jpg')):
                    if debug:
                        display(f'Reading from {filename}')
                    im = Image.open(filename)
                    im = imageTransformation(im, keyItemDetail, debug)
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
                             outputFolder                     : str=None,
                             outputAllParsedFrames            : bool=False,
                             startFrame                       : int=None,
                             endFrame                         : int=None):
    if fileType == constants.FileType.VIDEO:
        return getFramesFromVideo(
                    pathToVideo   = videoOrImageToParsePath,
                    fps           = fpsOfInputVideo,
                    everyXSeconds = processEveryXSecondsFromVideo,
                    outputFolder  = outputFolder,
                    outputAllParsedFrames = outputAllParsedFrames
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
                outputFolder                    = parseConfig.outputFolder,
                outputAllParsedFrames           = parseConfig.outputAllParsedFrames
            )

def saveVideoUsingParsingConfig(video:VideoFileClip, fileName:str, parseConfig:ParsingConfig):
    saveVideo(video, outputPath, codec=parseConfig.codec, threads=parseConfig.threads, bitrate=parseConfig.saveVideoAtBitrate)

def drawBoxes(im:Image, keyItems:list[KeyItemDetail], outlineColor:str='blue') -> Image:
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    for keyItem in keyItems:
        draw.rectangle(keyItem.coords.box(), outline=outlineColor)
    return im

def imageTransformationForScore(im:np.array) -> Image:
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

def imageTransformation(im:Image, type:KeyItemDetail, debug:bool=False) -> Image:
    cv2Im = np.array(im)
    if type.hsvFilters is not None:
        lower = np.array(type.hsvFilters[0])
        upper = np.array(type.hsvFilters[1])
        cv2Im = cv2.inRange(cv2Im, lower, upper)
    if type.keyItem in {constants.KeyItem.SCORE}:
        return imageTransformationForScore(cv2Im)
    elif type.keyItem in {constants.KeyItem.TIME}:
        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(cv2Im,kernel,iterations = 1)
        im1 = Image.fromarray(erosion)
        return im1
    elif type.keyItem in {constants.KeyItem.TEAM_NAME, constants.KeyItem.SCOREBOARD_PASSES_CHECK}:
        tmp = cv2.cvtColor(cv2Im, cv2.COLOR_BGR2GRAY)
        (th, newimg) = cv2.threshold(tmp, 50, 255, cv2.THRESH_BINARY)
        return Image.fromarray(newimg)
    else:
        if debug:
            print('No transformation found')
        return Image.fromarray(cv2Im)

@dataclass_json
@dataclass
class ParsingResult:
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
                return gameSettings.time() - time - timedelta(seconds=1) # these game phases report time as how much time has passed, add 1 second to deal with rounding.
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
        
    def to_dict(self):
        return {
            'gamePhase': self.gamePhase,
            'gameSettings': self.gameSettings.to_dict(),
            'time': self.time,
            'parsingResults': {k:v for k,v in self.parsingResults.items()}
        }
    
    def to_json(self):
        return json.dump(self.to_dict())
    
    @classmethod
    def from_json(jsonStr:str):
        d = json.loads(jsonStr)
        return ParsedGame(gameName=d['gameName'], gameSettings=GameSettings.from_dict(d['gameSettings']), time=float(d['time']), parsingResults={k:ParsingResult.from_dict(v) for k,v in d['parsingResults']})

@dataclass  
class ParsedGame:
    gameName:str
    keyMoments:dict
    parsedFrames:list[ParsedImage]

    def getStartTime(self) -> int:
        if self.parsedFrames:
            return self.parsedFrames[0].time
        else:
            return None

    def getEndTime(self) -> int:
        if self.parsedFrames:
            return self.parsedFrames[-1].time
        else:
            return None
    
    def to_dict(self):
        return {
            'gameName': self.gameName,
            'keyMoments': {k:[i.to_dict() for i in v] for k,v in self.keyMoments.items()}
        }
    
    def to_json(self):
        return json.dumps(self.to_dict())



class Timeline:
    '''
        Represents the timeline of a video or an individual image.
    '''
    def __init__(self, parsedFrames:list[ParsedImage], parseConfig:ParsingConfig, skipInitializing:bool=False):
        """_summary_

        Args:
            parsedFrames (list[ParsedImage]): a list of the parsed images from the video
            parseConfig (ParsingConfig): the parsing config to reference when re-constructing the timeline.
            skipInitializing (bool, optional): useful if you want to **not** immediately use the timeline and merge it at a later point.
        """
        self.parseConfig=parseConfig
        if skipInitializing:
            return
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
        for idx, frame in tqdm(enumerate(parsedFrames), desc='Generating timeline...'):
            if not frame or frame.gamePhase == constants.GamePhase.UNKNOWN:
                continue
            currentTime = frame.getValue(constants.KeyItem.TIME, constants.GameSide.NONE)
            if (lastKnownTime is None and currentTime is not None)\
                or (lastKnownTime is not None and currentTime is not None and (lastKnownTime + BUFFER < currentTime) and not(currentTime >= GOLDEN_GOAL_TIME - BUFFER and not(currentTime >= self.parseConfig.gameSettings.time() - BUFFER))):              
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
                if frame.gamePhase == constants.GamePhase.END_GAME_SCOREBOARD: #end game scoreboard doesn't have any TIME value, so instead we only keep the first instance of a scoreboard frame.
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
                time = self.parseConfig.gameSettings.time().total_seconds()-frame.getValue(constants.KeyItem.TIME, constants.GameSide.NONE).total_seconds()
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
        for gameNo,game in self.games.items():
            quality[gameNo] = self._evaluateGame(game)
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

    
    def gamesAsDicts(self) -> list[dict]:
        return [g.to_dict() for g in self.games.values()]

    
    def to_dict(self) -> dict:
        return {
            'parseConfig': self.parseConfig,
            'games': {k:g.to_dict() for k,g in self.games.items()}
        }
    
    def to_json(self):
        return json.dumps(self.to_dict())
    
def mergeTimelines(timelines:list[Timeline], parseConfig:ParsingConfig):
    """_summary_

    Args:
        timelines (list[Timeline]): must be in sorted order, the mergeTimelines will merge these in order.
    """
    allParsedFrames = []
    [allParsedFrames.extend(t.parsedImages) for t in timelines]
    return Timeline(parsedFrames=allParsedFrames, parseConfig=parseConfig, skipInitializing=False)

def runOcrOnImage(im:Image, keyItemDetail:KeyItemDetail, debug:bool=False, storedImageCache:ImageStore=None) -> str:
    if storedImageCache:
        if debug:
            print(f'[Ocr | Stored image] Calling stored image cache to lookup image...')
        value, dist = storedImageCache.lookupImageValue(im, keyItemDetail.maximumDistanceForStoredImages, debug=debug)
        if value is not None:
            if debug:
                print(f'[Ocr | Stored image] Found item in STORED IMAGES with distance of `{dist}` and value of `{value}`')
            return value, [dist]
    with tesserocr.PyTessBaseAPI(path=TESSERDATA_PATH) as api:
        api.SetPageSegMode(tesserocr.PSM.SINGLE_LINE)
        for option,value in keyItemDetail.tesserocrOptions.items():
            api.SetVariable(option, value)
        if keyItemDetail.numbersOnly:
            api.SetVariable("tessedit_char_whitelist", "0123456789")
        api.SetImage(im)
        res = api.GetUTF8Text()
        return re.sub(r'\n', '', res), api.AllWordConfidences()

def tryParseKeyItem(im:Image, gamePhaseDetail:GamePhaseDetail, keyItem:KeyItemDetail, gameSettings:GameSettings, debug:bool=False, ignoreErrors:bool=False,  storedImageCache:ImageStore=None, displayImages:bool=False) -> ParsingResult:
    box = keyItem.coords.box()
    cropped = im.crop(box)
    if debug:
        print(f'[{gamePhaseDetail.gamePhase}]: Outputting cropped image for {keyItem.keyItem} side {keyItem.side}, box={box}')
        display(cropped)
    transformed = None
    try:
        transformed = imageTransformation(cropped, keyItem, debug) if keyItem.shouldApplyImageTransformation else cropped
    except Exception as err:
        print(f'[{gamePhaseDetail.gamePhase}]:  Exception encountered when trying to transform image for {keyItem.keyItem} side {keyItem.side}')
        raise err
    parsedVal, confidences = runOcrOnImage(transformed, keyItem, storedImageCache=storedImageCache, debug=debug)
    parsingResult = ParsingResult(parsedValue=parsedVal, confidences=confidences)
    if debug:
        display(f'[{gamePhaseDetail.gamePhase}]:  ParsedValue: `{parsedVal}`')
        display(f'[{gamePhaseDetail.gamePhase}]:  Confidences: {confidences}')
        if displayImages:
            display(f'[{gamePhaseDetail.gamePhase}]:  Image parsed:')
            display(transformed)
    return parsingResult

def tryParseImageUsingDefaultSettings(im:Image, gamePhase:constants.GamePhase, gameSettings:GameSettings=None, debug:bool=False, ignoreErrors:bool=False,  imageCacheFolder:str=None, displayImages:bool=False, useConfigAsBackup:bool=True) -> ParsedImage:
    details = PARSE_CONFIG.gamePhaseDetails.getGamePhaseDetail(gamePhase)
    storedImageCache = None
    tmpDetails = GamePhaseDetails([details])
    if imageCacheFolder:
        storedImageCache = ImageStore(imageCacheFolder, tmpDetails, debug=debug)
    elif useConfigAsBackup and PARSE_CONFIG.imageCacheFolderPath:
        storedImageCache = ImageStore(PARSE_CONFIG.imageCacheFolderPath, tmpDetails, debug=debug)
    if not gameSettings and useConfigAsBackup:
        gameSettings = PARSE_CONFIG.gameSettings
    return tryParseImage(im, None, details, gameSettings=gameSettings, debug=debug, storedImageCache=storedImageCache)
        
def tryParseImage(im:Image, time:float, gamePhaseDetail:GamePhaseDetail, gameSettings:GameSettings, debug:bool=False, ignoreErrors:bool=False,  storedImageCache:ImageStore=None, displayImages:bool=False) -> ParsedImage:
    parsedImage = ParsedImage(gamePhaseDetail.gamePhase, im, time, gameSettings=gameSettings)
    if debug:
        display(f"[{gamePhaseDetail.gamePhase}]:  Extracting the following areas from the image:")
        display(drawBoxes(im.copy(), gamePhaseDetail.getKeyItemDetails()))
        #tmpIm = drawBoxes(im.copy(), gamePhaseDetail.getKeyItemDetails())
        #tmpIm.save('tmp_boxes.jpg')
    for keyItem in gamePhaseDetail.getKeyItemDetails():
        parsingResult = tryParseKeyItem(im, gamePhaseDetail, keyItem, gameSettings, debug, ignoreErrors, storedImageCache, displayImages)

        typedValue = None
        try:
            typedValue = parsingResult.parseToActualValue(gamePhase=gamePhaseDetail.gamePhase, keyItem=keyItem.keyItem, gameSettings=gameSettings)
        except Exception as err:
            if debug:
                print(f'[{gamePhaseDetail.gamePhase}]: Exception when parsing {keyItem.keyItem}. Text=`{parsingResult.parsedValue}`')
            if gamePhaseDetail.identifyingKeyItem[0] == keyItem.keyItem and gamePhaseDetail.identifyingKeyItem[1] == keyItem.side:
                print(f'[{gamePhaseDetail.gamePhase}]:  Was not {gamePhaseDetail.gamePhase} because {keyItem.keyItem} could not be parsed. Text=`{parsingResult.parsedValue}`')
                print(f'[{gamePhaseDetail.gamePhase}]:  {err}')
                return None, False       
            if keyItem.keyItem == constants.KeyItem.SCORE and gamePhaseDetail.gamePhase == constants.GamePhase.END_GAME_SCOREBOARD:
                if debug:
                    print('Retrying SCORE for scoreboard with no transformation...')
                keyItem.shouldApplyImageTransformation = False
                parsingResult = tryParseKeyItem(im, gamePhaseDetail, keyItem, gameSettings, debug, ignoreErrors, storedImageCache, displayImages)
        if gamePhaseDetail.gamePhase == constants.GamePhase.END_GAME_SCOREBOARD \
            and gamePhaseDetail.identifyingKeyItem[0] == keyItem.keyItem \
            and gamePhaseDetail.identifyingKeyItem[1] == keyItem.side \
            and typedValue != constants.SCOREBOARD_PASSES_CHECK_VALUE:
            if debug:
                print(f'[{gamePhaseDetail.gamePhase}]:  Was not {gamePhaseDetail.gamePhase} because {keyItem.keyItem} could not be parsed. Text=`{parsingResult.parsedValue}`')
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
    for frame, time in tqdm(zip(frames, times), total=len(frames), desc='Processing frames...'):
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
    totalFrames = getTotalFramesFromVideo(parseConfig.videoOrImageToParsePath)
    cpus = parseConfig.cpus if parseConfig.cpus else multiprocessing.cpu_count
    chunkSize = max(parseConfig.minChunkSizeInFrames, totalFrames // cpus )
    frameChunks = [[i, i+chunkSize] for i in range(0, totalFrames, chunkSize)]  # split the frames into chunk lists
    frameChunks[-1][-1] = min(frameChunks[-1][-1], totalFrames-1)  # make sure last chunk has correct end frame, also handles case chunk_size < total
    if parseConfig.imageCacheFolderPath is not None:
        storedImageCache = ImageStore(parseConfig.imageCacheFolderPath, parseConfig.gamePhaseDetails)
    with multiprocessing.ProcessPoolExecutor(max_workers=cpus) as executor:
        futures = [executor.submit(parseFrames, parseConfig, f[0], f[1], storedImageCache)
                   for f in frameChunks]  # submit the processes: extract_frames(...)
        with tqdm(total=len(frameChunks)-1) as pbar:
            for i, f in enumerate(multiprocessing.as_completed(futures)):  # as each process completes
                pbar.update(1)
    debugResults = []
    timelines = []
    for f in futures:
        timeline, debugResult = f.result()
        timelines.append(timeline)
        debugResults.append(debugResult)
    mergedTimeline = mergeTimelines(timelines, parseConfig)
    return mergedTimeline, debugResults

def parseFrames(parseConfig:ParsingConfig, startFrame:int, endFrame:int, imageStore:ImageStore):
    print(f'Getting frames from file...')
    frames, times = getFramesFromFileContentUsingParsingConfig(parseConfig)
    print(f'Successfully got frames from the file `{parseConfig.videoOrImageToParsePath}`')
    results = process(frames, times, parseConfig.gamePhaseDetails, parseConfig.gameSettings, parseConfig.debug, parseConfig.ignoreParsingErrors, storedImageCache=imageStore)
    timeline = Timeline(parsedFrames=results, parseConfig=parseConfig, skipInitializing=True)
    if parseConfig.debug:
        return timeline, results
    return timeline, None

def split(timeline:Timeline, parseConfig:ParsingConfig):
    outputBasePath = os.path.join(parseConfig.outputFolder, 'videos')
    with VideoFileClip(parseConfig.videoOrImageToParsePath) as video:
        for gameKey, parsedGame in timeline.games.items():
            outputGamesPath = os.path.join(outputBasePath, 'games', parsedGame.name)
            Path(outputGamesPath).mkdir(parents=True, exist_ok=True)
            if parseConfig.outputSplitGames:
                startTime = (parsedGame.getStartTime() / 60.0) - parseConfig.bufferTimeBeforeGamesInSeconds
                endTime = (parsedGame.getEndTime() / 60.0) - parseConfig.bufferTimeAfterGamesInSeconds
                clips = extractClips(video, [(startTime, endTime)])
                [saveVideoUsingParsingConfig(clip, os.path.join(outputGamesPath, f"{parsedGame.name}.mp4"), parseConfig) for clip in clips]
            if parseConfig.outputSplitGoals:
                outputGoalsPath = os.path.join(outputGamesPath, 'goals')
                Path(outputGoalsPath).mkdir(parents=True, exist_ok=True)
                goalTimes = []
                for goalMoment in [constants.GamePhase.GOAL_SCORED_LEFT_HAND_SIDE, constants.GamePhase.GOAL_SCORED_RIGHT_HAND_SIDE]:
                    for frame in parsedGame.keyMoments[constants.GamePhase.GOAL_SCORED_LEFT_HAND_SIDE]:
                        goalTimes.append((frame.time / 60.0) - parseConfig.bufferTimeBeforeGoalsInSeconds, (frame.time / 60.0) - parseConfig.bufferTimeAfterGoalsInSeconds, goalMoment)
                goalTimes = sorted(goalTimes, key=lambda x: x[0])
                times = [(x[0], x[1]) for x in goalTimes]
                clips = extractClips(video, times)
                [saveVideoUsingParsingConfig(clip, os.path.join(outputGoalsPath, f"{parsedGame.name}.goal.{idx}.{str(goalTimes[idx][2])}.mp4"), parseConfig) for idx,clip in enumerate(clips)]



def run(parseConfig:ParsingConfig):
    timeline, frames = parse(parseConfig)
    split(timeline, parseConfig)