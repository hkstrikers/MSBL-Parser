from dataclasses import dataclass
import constants

@dataclass
class Coordinates:
    '''
        A class which represents a rectangular region of an image. 
    '''
    left:int
    upper:int
    right:int
    lower:int

    def flip(self,axis:str) -> Coordinates:
        if axis == 'y':
            return Coordinates(constants.Resolution.WIDTH-self.right, self.upper, constants.Resolution.WIDTH-self.left, self.lower)
        elif axis == 'x':
            return Coordinates(self.right, constants.Resolution.HEIGHT-self.lower, self.left, constants.Resolution.HEIGHT-self.upper)
        else:
            raise ValueError('Axis must be either `x` or `y`')
        return None


        
@dataclass
class KeyItemDetails:
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
    name:constants.KeyItem

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