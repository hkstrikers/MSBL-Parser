from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from datetime import timedelta
from typing import Callable
import src.constants as constants

@dataclass_json
@dataclass
class Coordinates:
    '''
        A class which represents a rectangular region of an image. 
    '''
    left              : int
    upper             : int
    right             : int
    lower             : int

    def flipAlongYAxis(self, resolution = constants.RESOLUTION):
        return Coordinates(resolution.WIDTH - self.right, self.upper, resolution.WIDTH - self.left, self.lower)

    def box(self) -> tuple:
        return (self.left, self.upper, self.right, self.lower)

@dataclass_json 
@dataclass
class KeyItemDetail:
    '''
        A class representing details of KeyItems.
    '''

    '''
        The rectangular region of the image where the item is located.
    '''
    coords:Coordinates

    '''
        The name of the key item.
    '''
    keyItem:constants.KeyItem

    '''
        The side represented by this item.
    '''
    side:constants.GameSide

    '''
        Indicates whether this KeyItem contains only numbers data or not.
        This is a useful hint to the OCR.
    '''
    numbersOnly:bool

    '''
        Other tesserocrOptions to be used with this KeyItem.
    '''
    tesserocrOptions:dict=field(default_factory=lambda: dict())

    '''
        If looking up stored images, the maximum distance away the image should be to be considered a match.
    '''
    maximumDistanceForStoredImages:int=field(default_factory=lambda: 10)

    '''
        The HSV filters to apply to this key item detail -- NULL means apply the default for this key item.
    '''
    hsvFilters:list[list]=field(default_factory=lambda: None)

    '''
        An override, indicating not to apply image transformation to this key item.
    '''
    shouldApplyImageTransformation:bool = True
      


@dataclass_json
@dataclass
class KeyItemDetails:
    keyItemDetails:list[KeyItemDetail]

    def __post_init__(self):
        self.dictLookup = {}
        for detail in self.keyItemDetails:
            if detail.keyItem not in self.dictLookup:
                self.dictLookup[detail.keyItem] = {}
            if detail.side not in self.dictLookup[detail.keyItem]:
                self.dictLookup[detail.keyItem][detail.side] = detail
            else:
                raise ValueError(f"Cannot provide multiple details for the same keyItem and side. keyItem=`{detail.keyItem}`, side=`{detail.side}`")
    
    def getKeyItemDetail(self, keyItem:constants.KeyItem, side:constants.GameSide) -> KeyItemDetail:
        if keyItem not in self.dictLookup or side not in self.dictLookup[keyItem]:
            return None
        return self.dictLookup[keyItem][side]
    
    def getKeyItems(self) -> list[constants.KeyItem]:
        return list(self.dictLookup.keys())
    
    def getKeyItemDetails(self) -> list[KeyItemDetail]:
        return self.keyItemDetails
    
    def getSides(self, keyItem:constants.KeyItem) -> list[constants.GameSide]:
        if keyItem in self.dictLookup:
            return list(self.dictLookup[keyItem].keys())
        else:
            return []

@dataclass_json
@dataclass
class GamePhaseDetail:
    gamePhase         : constants.GamePhase
    keyItemDetails    : KeyItemDetails
    identifyingKeyItem: tuple

    def getKeyItemDetail(self, keyItem:constants.KeyItem, side:constants.GameSide) -> KeyItemDetail:
        return self.keyItemDetails.getKeyItemDetail(keyItem, side)

    def getKeyItemDetails(self) -> list[KeyItemDetail]:
        return self.keyItemDetails.getKeyItemDetails()

@dataclass_json
@dataclass
class GamePhaseDetails:
    gamePhaseDetails  : list[GamePhaseDetail]

    def __post_init__(self):
        self.dictLookup = {}
        for detail in self.gamePhaseDetails:
            if detail.gamePhase not in self.dictLookup:
                self.dictLookup[detail.gamePhase] = detail
            else:
                raise ValueError(f"Cannot provide multiple details for the same game phase. gamePhase=`{detail.gamePhase}`")

    
    def getGamePhaseDetail(self, gamePhase:constants.GamePhase) -> GamePhaseDetail:        
        if gamePhase not in self.dictLookup:
            return None
        return self.dictLookup[gamePhase]
    
    def getKeyItemDetail(self, gamePhase:constants.GamePhase, keyItem:constants.KeyItem, side:constants.GameSide) -> KeyItemDetail:
        detail = self.getGamePhaseDetail(gamePhase)
        if detail:
            return detail.getKeyItemDetail(keyItem, side)
        return detail
    
    def getKeyItemDetails(self, gamePhase:constants.GamePhase) -> list[KeyItemDetail]:
        detail = self.getGamePhaseDetail(gamePhase)
        if detail:
            return detail.keyItemDetails.getKeyItemDetails()
        return detail
    
    def getGamePhases(self) -> list[constants.GamePhase]:
        return list(self.dictLookup.keys())
    
    def getGamePhaseDetails(self) -> list[GamePhaseDetail]:
        return self.gamePhaseDetails

@dataclass_json
@dataclass
class GameSettings:
    time        : timedelta = timedelta(minutes=4)

@dataclass_json
@dataclass
class ParsingConfig:
    videoOrImageToParsePath         : str
    FileType                        : constants.FileType
    gamePhaseDetails                : GamePhaseDetails
    imageCacheFolderPath            : str = None
    resolution                      : constants.DefaultResolution = constants.RESOLUTION
    processEveryXSecondsFromVideo   : int = 4
    fpsOfInputVideo                 : int = 60
    outputFolder                    : str=None
    debug                           : bool=False
    ignoreParsingErrors             : bool=False
    gameSettings                    : GameSettings=GameSettings()