from enum import Enum
from dataclasses import dataclass,field

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
        Phase of the game where a side has just scored.
    '''
    GOAL_SCORED = 'GOAL_SCORED'

    '''
        Phase of the game after the game has ended.
    '''
    IN_GAME_FINAL_RESULT = 'IN_GAME_FINAL_RESULT'

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

class GameSide(str, Enum):
    '''
        Represents whether a KeyItem is associated with the left-hand side, right-hand side, or none (e.g., the time doesn't belong to a side).
    '''
    LEFT    = 'LEFT_HAND_SIDE'
    RIGHT   = 'RIGHT_HAND_SIDE'
    NONE    = 'NONE'

class DefaultResolution:
    HEIGHT    : int = 1080
    WIDTH     : int = 1920

class FileType(str,Enum):
    VIDEO = 'VIDEO'
    IMAGE = 'IMAGE'

RESOLUTION = DefaultResolution()