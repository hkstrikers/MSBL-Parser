from enum import Enum
from dataclasses import dataclass,field

from dataclasses_json import dataclass_json

class GamePhase(str, Enum):
    '''
        Represents a phase of a game, each moment is made up of one or more KeyItem's.
    '''

    '''
        Phase of the game where players are actively controlling their characters
        OR the scoreboard and time are shown at the top of the screen (e.g., during kickoff).
    '''
    IN_GAME = 'IN_GAME'

    '''
        Phase of the game where the left hand side has just scored.
    '''
    GOAL_SCORED_LEFT_HAND_SIDE = 'GOAL_SCORED_LEFT_HAND_SIDE'

    '''
        Phase of the game where the right hand side has just scored.
    '''
    GOAL_SCORED_RIGHT_HAND_SIDE = 'GOAL_SCORED_RIGHT_HAND_SIDE'

    '''
        Phase of the game after the game has ended and left hand side won
    '''
    IN_GAME_FINAL_RESULT_LEFT_HAND_SIDE = 'IN_GAME_FINAL_RESULT_LEFT_HAND_SIDE'

    '''
        Phase of the game after the game has ended and right hand side won
    '''
    IN_GAME_FINAL_RESULT_RIGHT_HAND_SIDE = 'IN_GAME_FINAL_RESULT_RIGHT_HAND_SIDE'

    '''
        Phase of the game showing the scoreboard after the game has finished.
    '''
    END_GAME_SCOREBOARD = 'SCOREBOARD'


    '''
        DEFAULT GAME PHASE
    '''
    UNKNOWN = 'UKNOWN'


class KeyItem(str, Enum):
    '''
        Represents a key piece of data presented on the screen during a GamePhase.
    '''

    '''
        Represents the time remaining in a game.
    '''
    TIME = 'TIME'

    '''
        Represents the team name of one side in a game.
    '''
    TEAM_NAME = 'TEAM_NAME'

    '''
        Represents the score of one side in a game.
    '''
    SCORE = 'SCORE'

    '''
        Represents the text the flavor text the game presents during a 'GOAL_SCORED*' phase or during a 'IN_GAME_FINAL_RESULT*' phase.
    '''
    FLAVOR_TEXT = 'FLAVOR_TEXT'
    
    SHOTS_ON_GOAL = 'SHOTS_ON_GOAL'
    
    HYPER_STRIKES = 'HYPER_STRIKES'
    
    ITEMS_USED = 'ITEMS_USED'
    
    TACKLES = 'TACKLES'
    
    PASSES = 'PASSES'
    
    INTERCEPTIONS = 'INTERCEPTIONS'
    
    ASSISTS = 'ASSISTS'
    
    POSESSION = 'POSESSION'
    
    WINNER = 'WINNER'
    SCOREBOARD_PASSES_CHECK = 'SCOREBOARD_PASSES_CHECK'
    SCORE_SCOREBOARD_RETRY = 'SCORE_SCOREBOARD_RETRY'

SCOREBOARD_PASSES_CHECK_VALUE = 'PASSES'

SCOREBOARD_STATS = [
    KeyItem.SHOTS_ON_GOAL,
    KeyItem.HYPER_STRIKES,
    KeyItem.ITEMS_USED,
    KeyItem.TACKLES,
    KeyItem.PASSES,
    KeyItem.INTERCEPTIONS,
    KeyItem.ASSISTS,
    KeyItem.POSESSION
]

class GameSide(str, Enum):
    '''
        Represents whether a KeyItem is associated with the left-hand side, right-hand side, or none (e.g., the time doesn't belong to a side).
    '''
    LEFT    = 'LEFT'
    RIGHT   = 'RIGHT'
    NONE    = 'NONE'

@dataclass_json
@dataclass
class DefaultResolution:
    HEIGHT    : int = 1080
    WIDTH     : int = 1920

class FileType(str,Enum):
    VIDEO = 'VIDEO'
    IMAGE = 'IMAGE'

RESOLUTION = DefaultResolution()  