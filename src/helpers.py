from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
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
    sideRepresented   : constants.GameSide

    def flip(self):
        if self.sideRepresented in [constants.GameSide.LEFT, constants.GameSide.RIGHT]:
            return Coordinates(constants.RESOLUTION.WIDTH - self.right, constants.RESOLUTION.WIDTH - self.lower, self.left, self.upper)
        else:
            raise ValueError(f'`sideRepresented`=`{self.sideRepresented}` cannot be flipped.')
        return None

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
    maximumDistanceForStoredImages:int=field(default_factory=lambda: 0)
      
@dataclass_json
@dataclass
class GamePhaseDetail:
    gamePhase         : constants.GamePhase
    keyItemDetails    : list[KeyItemDetail]
    identifyingKeyItem: KeyItemDetail

@dataclass_json
@dataclass
class ParsingConfig:
    videoOrImageToParsePath         : str
    FileType                        : constants.FileType
    gamePhaseDetails                : list[GamePhaseDetail]
    imageCacheFolderPath            : str
    resolution                      : constants.DefaultResolution = constants.RESOLUTION
    processEveryXSecondsFromVideo   : int = 4
    fpsOfInputVideo                 : int = 60
    outputFolder                    : str=None
    debug                           : bool=False