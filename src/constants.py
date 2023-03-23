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
        Phase of the game where the player playing on the left-hand side of the field has just scored.
    '''
    GOAL_SCORED = 'GOAL_SCORE_LEFT_HAND_SIDE'

    '''
        Phase of the game where the player playing on the right-hand side of the field has just scored.
    '''
    GOAL_SCORE_RIGHT_HAND_SIDE = 'GOAL_SCORE_RIGHT_HAND_SIDE'

    '''
        Phase of the game when the player playing on the left-hand side of the field has won.
    '''
    IN_GAME_FINAL_RESULT_LEFT_HAND_SIDE = 'IN_GAME_FINAL_RESULT_LEFT_HAND_SIDE'

    '''
        Phase of the game when the player playing on the right-hand side of the field has won.
    '''
    IN_GAME_FINAL_RESULT_RIGHT_HAND_SIDE = 'IN_GAME_FINAL_RESULT_RIGHT_HAND_SIDE'

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
    LEFT = 'LEFT_HAND_SIDE'
    RIGHT = 'RIGHT_HAND_SIDE'
    NONE = 'NONE'

class Resolution(Enum):
    HEIGHT = 1080
    WIDTH = 1920

BOUNDING_BOXES = {
    'IN_GAME' :
    {
        'Time': Item((875,0,1040,65), 'Time', ['none'], False),
        'TeamName': Item((185,85,400,125),'TeamName', ['left','right'], False),
        'Score': Item((700,20,825,82),'Score', ['left', 'right'], True, tesserocrOptions={'PageSegMode': tesserocr.PSM.SINGLE_WORD})    
    },
    'GOAL_SCORED_LEFT_HAND_SIDE' :
    {
        'Time': Item((250,850,380,900),'Time', ['none'], False),
        'ScoreLeft': Item((1100,735,1330,900), 'ScoreLeft', ['none'], True),
        'ScoreRight': Item((1450,735,1680,900),'ScoreRight', ['none'], True),
        'TeamName':  Item((490,850,900,900),'TeamName', ['none'], False),
        'FlavorText': Item((250,925,1000,1000),'ScoreText', ['none'], False)
    },
    'IN_GAME_FINAL_RESULT_LEFT_HAND_SIDE' :
    {
        'ScoreLeft': Item((1100,735,1330,900), 'ScoreLeft', ['none'], True),
        'ScoreRight': Item((1450,735,1680,900),'ScoreRight', ['none'], True),
        'TeamName':  Item((330,850,700,900),'TeamName', ['none'], False),
        'FlavorText': Item((250,925,1000,1000),'ScoreText', ['none'], False)
    }, 
}