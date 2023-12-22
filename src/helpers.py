from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from datetime import timedelta
from typing import Callable
import constants as constants

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
    identifyingKeyItem: list[str]

    def getKeyItemDetail(self, keyItem:constants.KeyItem, side:constants.GameSide) -> KeyItemDetail:
        return self.keyItemDetails.getKeyItemDetail(keyItem, side)

    def getKeyItemDetails(self) -> list[KeyItemDetail]:
        return self.keyItemDetails.getKeyItemDetails()

@dataclass_json
@dataclass
class GamePhaseDetails:
    details  : list[GamePhaseDetail]

    def __post_init__(self):
        self.dictLookup = {}
        for detail in self.details:
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
        return self.details

@dataclass_json
@dataclass
class GameSettings:
    timeInMinutes        : int = 4
    
    def time(self) -> timedelta:
        return timedelta(minutes=self.timeInMinutes)



DEFAULT_GAME_PHASE_DETAILS = GamePhaseDetails([
    GamePhaseDetail(
        gamePhase=constants.GamePhase.IN_GAME,
        keyItemDetails= KeyItemDetails([
            KeyItemDetail(
                Coordinates(875,0,1040,65),
                constants.KeyItem.TIME,
                constants.GameSide.NONE,
                numbersOnly=False,
                tesserocrOptions={}
            ),
            KeyItemDetail(
                Coordinates(185,85,400,125),
                constants.KeyItem.TEAM_NAME,
                constants.GameSide.LEFT,
                numbersOnly=False,
                tesserocrOptions={}
            ),
            KeyItemDetail(
                Coordinates(185,85,400,125).flipAlongYAxis(),
                constants.KeyItem.TEAM_NAME,
                constants.GameSide.RIGHT,
                numbersOnly=False,
                tesserocrOptions={}
            ),
            KeyItemDetail(
                Coordinates(700,20,825,84),
                constants.KeyItem.SCORE,
                constants.GameSide.LEFT,
                numbersOnly=True,
                tesserocrOptions={}
            ),
            KeyItemDetail(
                Coordinates(700,20,825,84).flipAlongYAxis(),
                constants.KeyItem.SCORE,
                constants.GameSide.RIGHT,
                numbersOnly=True,
                tesserocrOptions={}
            ),
        ]),
        identifyingKeyItem=(constants.KeyItem.TIME, constants.GameSide.NONE)
    ),
    GamePhaseDetail(
        gamePhase=constants.GamePhase.GOAL_SCORED_LEFT_HAND_SIDE,
        keyItemDetails= KeyItemDetails([
            KeyItemDetail(
                Coordinates(250,850,380,900),
                constants.KeyItem.TIME,
                constants.GameSide.NONE,
                numbersOnly=False,
                tesserocrOptions={}
            ),
            KeyItemDetail(
                Coordinates(1100,735,1330,900),
                constants.KeyItem.SCORE,
                constants.GameSide.LEFT,
                numbersOnly=True,
                tesserocrOptions={}
            ),
            KeyItemDetail(
                Coordinates(1450,735,1680,900),
                constants.KeyItem.SCORE,
                constants.GameSide.RIGHT,
                numbersOnly=True,
                tesserocrOptions={}
            ),
            KeyItemDetail(
                Coordinates(490,850,900,900),
                constants.KeyItem.TEAM_NAME,
                constants.GameSide.LEFT,
                numbersOnly=False,
                tesserocrOptions={}
            ),
            KeyItemDetail(
                Coordinates(250,925,1000,1000),
                constants.KeyItem.FLAVOR_TEXT,
                constants.GameSide.NONE,
                numbersOnly=True,
                tesserocrOptions={}
            )
        ]),
        identifyingKeyItem=(constants.KeyItem.TIME, constants.GameSide.NONE)
    ),
    GamePhaseDetail(
        gamePhase=constants.GamePhase.GOAL_SCORED_RIGHT_HAND_SIDE,
        keyItemDetails= KeyItemDetails([
            KeyItemDetail(
                Coordinates(250,850,380,900).flipAlongYAxis(),
                constants.KeyItem.TIME,
                constants.GameSide.NONE,
                numbersOnly=False,
                tesserocrOptions={}
            ),
            KeyItemDetail(
                Coordinates(1100,735,1330,900).flipAlongYAxis(),
                constants.KeyItem.SCORE,
                constants.GameSide.LEFT,
                numbersOnly=True,
                tesserocrOptions={}
            ),
            KeyItemDetail(
                Coordinates(1450,735,1680,900).flipAlongYAxis(),
                constants.KeyItem.SCORE,
                constants.GameSide.RIGHT,
                numbersOnly=True,
                tesserocrOptions={}
            ),
            KeyItemDetail(
                Coordinates(490,850,900,900).flipAlongYAxis(),
                constants.KeyItem.TEAM_NAME,
                constants.GameSide.RIGHT,
                numbersOnly=False,
                tesserocrOptions={}
            ),
            KeyItemDetail(
                Coordinates(250,925,1000,1000).flipAlongYAxis(),
                constants.KeyItem.FLAVOR_TEXT,
                constants.GameSide.NONE,
                numbersOnly=True,
                tesserocrOptions={}
            )
        ]),
        identifyingKeyItem=(constants.KeyItem.TIME, constants.GameSide.NONE)
    ),
    GamePhaseDetail(
        gamePhase=constants.GamePhase.IN_GAME_FINAL_RESULT_LEFT_HAND_SIDE,
        keyItemDetails= KeyItemDetails([
            KeyItemDetail(
                Coordinates(1100,735,1330,900),
                constants.KeyItem.SCORE,
                constants.GameSide.LEFT,
                numbersOnly=True,
                tesserocrOptions={}
            ),
            KeyItemDetail(
                Coordinates(330,850,700,900),
                constants.KeyItem.TEAM_NAME,
                constants.GameSide.LEFT,
                numbersOnly=False,
                tesserocrOptions={}
            ),
            KeyItemDetail(
                Coordinates(1450,735,1680,900),
                constants.KeyItem.SCORE,
                constants.GameSide.RIGHT,
                numbersOnly=True,
                tesserocrOptions={}
            ),
            KeyItemDetail(
                Coordinates(250,925,1000,1000),
                constants.KeyItem.FLAVOR_TEXT,
                constants.GameSide.NONE,
                numbersOnly=True,
                tesserocrOptions={}
            )
        ]),
        identifyingKeyItem=(constants.KeyItem.SCORE, constants.GameSide.LEFT)
    ),
    GamePhaseDetail(
        gamePhase=constants.GamePhase.IN_GAME_FINAL_RESULT_RIGHT_HAND_SIDE,
        keyItemDetails= KeyItemDetails([
                    KeyItemDetail(
                        Coordinates(1100,735,1330,900).flipAlongYAxis(),
                        constants.KeyItem.SCORE,
                        constants.GameSide.LEFT,
                        numbersOnly=True,
                        tesserocrOptions={}
                    ),
                    KeyItemDetail(
                        Coordinates(1450,735,1680,900).flipAlongYAxis(),
                        constants.KeyItem.SCORE,
                        constants.GameSide.RIGHT,
                        numbersOnly=True,
                        tesserocrOptions={}
                    ),
                KeyItemDetail(
                    Coordinates(330,850,700,900).flipAlongYAxis(),
                    constants.KeyItem.TEAM_NAME,
                    constants.GameSide.RIGHT,
                    numbersOnly=False,
                    tesserocrOptions={}
                ),
                KeyItemDetail(
                    Coordinates(250,925,1000,1000).flipAlongYAxis(),
                    constants.KeyItem.FLAVOR_TEXT,
                    constants.GameSide.NONE,
                    numbersOnly=True,
                    tesserocrOptions={}
                )
            
    ]),
        identifyingKeyItem=(constants.KeyItem.SCORE, constants.GameSide.LEFT)
    ),
    GamePhaseDetail(
        gamePhase=constants.GamePhase.END_GAME_SCOREBOARD,
        keyItemDetails= KeyItemDetails( [
            KeyItemDetail(coords=Coordinates(left=890, upper=705, right=1030, lower=750), keyItem=constants.KeyItem.SCOREBOARD_PASSES_CHECK, side=constants.GameSide.NONE, numbersOnly=False, tesserocrOptions={"tessedit_char_whitelist": "PASE"}),
            KeyItemDetail(coords=Coordinates(left=600, upper=100, right=850, lower=315), keyItem=constants.KeyItem.SCORE, side=constants.GameSide.LEFT, numbersOnly=True, tesserocrOptions={}),
            KeyItemDetail(coords=Coordinates(left=225, upper=25, right=600, lower=60), keyItem=constants.KeyItem.TEAM_NAME, side=constants.GameSide.LEFT, numbersOnly=False, tesserocrOptions={}),
            KeyItemDetail(coords=Coordinates(left=502, upper=410, right=598, lower=459), keyItem=constants.KeyItem.SHOTS_ON_GOAL, side=constants.GameSide.LEFT, numbersOnly=True, tesserocrOptions={}),
            KeyItemDetail(coords=Coordinates(left=502, upper=485, right=598, lower=533), keyItem=constants.KeyItem.HYPER_STRIKES, side=constants.GameSide.LEFT, numbersOnly=True, tesserocrOptions={}),
            KeyItemDetail(coords=Coordinates(left=502, upper=554, right=598, lower=607), keyItem=constants.KeyItem.ITEMS_USED, side=constants.GameSide.LEFT, numbersOnly=True, tesserocrOptions={}),
            KeyItemDetail(coords=Coordinates(left=502, upper=628, right=598, lower=681), keyItem=constants.KeyItem.TACKLES, side=constants.GameSide.LEFT, numbersOnly=True, tesserocrOptions={}),
            KeyItemDetail(coords=Coordinates(left=502, upper=698, right=598, lower=750), keyItem=constants.KeyItem.PASSES, side=constants.GameSide.LEFT, numbersOnly=True, tesserocrOptions={}),
            KeyItemDetail(coords=Coordinates(left=502, upper=770, right=598, lower=820), keyItem=constants.KeyItem.INTERCEPTIONS, side=constants.GameSide.LEFT, numbersOnly=True, tesserocrOptions={}),
            KeyItemDetail(coords=Coordinates(left=502, upper=845, right=598, lower=895), keyItem=constants.KeyItem.ASSISTS, side=constants.GameSide.LEFT, numbersOnly=True, tesserocrOptions={}),
            KeyItemDetail(coords=Coordinates(left=495, upper=915, right=556, lower=965), keyItem=constants.KeyItem.POSESSION, side=constants.GameSide.LEFT, numbersOnly=True, tesserocrOptions={}),
            KeyItemDetail(coords=Coordinates(left=1070, upper=100, right=1320, lower=315), keyItem=constants.KeyItem.SCORE, side=constants.GameSide.RIGHT, numbersOnly=True, tesserocrOptions={}),
            KeyItemDetail(coords=Coordinates(left=1320, upper=25, right=1695, lower=60), keyItem=constants.KeyItem.TEAM_NAME, side=constants.GameSide.RIGHT, numbersOnly=False, tesserocrOptions={}),
            KeyItemDetail(coords=Coordinates(left=1320, upper=410, right=1416, lower=459), keyItem=constants.KeyItem.SHOTS_ON_GOAL, side=constants.GameSide.RIGHT, numbersOnly=True, tesserocrOptions={}),
            KeyItemDetail(coords=Coordinates(left=1320, upper=485, right=1416, lower=533), keyItem=constants.KeyItem.HYPER_STRIKES, side=constants.GameSide.RIGHT, numbersOnly=True, tesserocrOptions={}),
            KeyItemDetail(coords=Coordinates(left=1320, upper=554, right=1416, lower=607), keyItem=constants.KeyItem.ITEMS_USED, side=constants.GameSide.RIGHT, numbersOnly=True, tesserocrOptions={}),
            KeyItemDetail(coords=Coordinates(left=1320, upper=628, right=1416, lower=681), keyItem=constants.KeyItem.TACKLES, side=constants.GameSide.RIGHT, numbersOnly=True, tesserocrOptions={}),
            KeyItemDetail(coords=Coordinates(left=1320, upper=698, right=1416, lower=750), keyItem=constants.KeyItem.PASSES, side=constants.GameSide.RIGHT, numbersOnly=True, tesserocrOptions={}),
            KeyItemDetail(coords=Coordinates(left=1320, upper=770, right=1416, lower=820), keyItem=constants.KeyItem.INTERCEPTIONS, side=constants.GameSide.RIGHT, numbersOnly=True, tesserocrOptions={}),
            KeyItemDetail(coords=Coordinates(left=1320, upper=845, right=1416, lower=895), keyItem=constants.KeyItem.ASSISTS, side=constants.GameSide.RIGHT, numbersOnly=True, tesserocrOptions={}),
            KeyItemDetail(coords=Coordinates(left=1320, upper=915, right=1379, lower=965), keyItem=constants.KeyItem.POSESSION, side=constants.GameSide.RIGHT, numbersOnly=True, tesserocrOptions={}),
            KeyItemDetail(coords=Coordinates(left=1980-1800, upper=156, right=1980-1480, lower=225), keyItem=constants.KeyItem.WINNER, side=constants.GameSide.LEFT, numbersOnly=False, tesserocrOptions={"tessedit_char_whitelist": "WINER!"}),
            KeyItemDetail(coords=Coordinates(left=1480, upper=156, right=1800, lower=225), keyItem=constants.KeyItem.WINNER, side=constants.GameSide.RIGHT, numbersOnly=False, tesserocrOptions={"tessedit_char_whitelist": "WINER!"})
        ]),
        identifyingKeyItem=(constants.KeyItem.SCOREBOARD_PASSES_CHECK, constants.GameSide.NONE)
    )
])

@dataclass_json
@dataclass
class ParsingConfig:
    videoOrImageToParsePath         : str
    FileType                        : constants.FileType
    gamePhaseDetails                : GamePhaseDetails = DEFAULT_GAME_PHASE_DETAILS
    imageCacheFolderPath            : str = None
    resolution                      : constants.DefaultResolution = constants.RESOLUTION
    processEveryXSecondsFromVideo   : int = 4
    fpsOfInputVideo                 : int = 60
    outputFolder                    : str=None
    debug                           : bool=False
    ignoreParsingErrors             : bool=False
    gameSettings                    : GameSettings=GameSettings()
    outputAllParsedFrames           : bool=False
    outputSplitGames                : bool=False
    outputSplitGoals                : bool=False
    bufferTimeBeforeGamesInSeconds  : int= 10
    bufferTimeAfterGamesInSeconds   : int= 10
    bufferTimeBeforeGoalsInSeconds  : int= 10
    bufferTimeAfterGoalsInSeconds   : int= 10
    outputVideoCodec                : str=None
    cpus                            : int= None
    chunksPerCpu                    : int=8
    minChunkSizeInFrames            : int=1000
    saveVideoAtBitrate              : str= "200K"
    codec                           : str=None