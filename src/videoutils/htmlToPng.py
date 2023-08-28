import asyncio
import datetime
import json
import sys
sys.path.append('../..')  # Add the parent directory to the Python path

from dataclasses import dataclass
import os
from dataclasses_json import dataclass_json
from pyppeteer import launch
from typing import TypedDict
from src import config
import http.server
import socketserver


CONFIG = config.CONFIG
print(CONFIG)
# Open and read the JSON file
TSH_BASE_PATH = CONFIG['videoConfig']['tournamentStreamHelperBasePath']
TSH_LAYOUTS_PATH = os.path.join(TSH_BASE_PATH, "layout")
TSH_LAYOUTS_PATH = os.path.join(TSH_BASE_PATH, "layout")
TSH_LAYOUTS_MSBL_PATH = os.path.join(TSH_LAYOUTS_PATH, "scoreboard_msbl", "index.html")
TSH_LAYOUTS_MSBL_MSL_PATH = os.path.join(TSH_LAYOUTS_PATH, "scoreboard_msbl_msl", "index.html")
TSH_OUT_PROGRAM_STATE_PATH = os.path.join(TSH_BASE_PATH, "out", "program_state.json")
TSH_ASSETS_PATH = os.path.join(TSH_BASE_PATH, 'assets')
TSH_COUNTRIES_PATH = os.path.join(TSH_ASSETS_PATH, 'countries+states+cities.json')
TSH_COUNTRIES_FLAGS_BASE_PATH = os.path.join(TSH_ASSETS_PATH, 'country_flag')

from dataclasses import dataclass
from typing import Dict, List, Union
from dataclasses_json import dataclass_json
from aiohttp import web
from datetime import time, timedelta
from pathlib import Path
import copy

'''
    TSH PROGRAM STATE CLASSES

    These are a tight-coupling with TournamentStreamHelper (TSH).
    Why?
    The custom JS used to power TSH can be re-used to create the exact same view as TSH.
    Otherwise, the visuals may look different.
'''
@dataclass_json
@dataclass
class Country:
    code: str
    asset: str=None
    display_name: str=None
    en_name: str=None
    latitude: str=None
    longitude: str=None
    name: str=None

@dataclass_json
@dataclass
class State:
    asset: str
    code: str
    latitude: str
    longitude: str
    name: str

@dataclass_json
@dataclass
class Character:
    name: str
    assets: dict
    skin: str

@dataclass_json
@dataclass
class Player:
    country: Country
    team: str
    name: str
    pronoun: str=None
    character: Dict[int, Character]=None
    real_name: str=None
    twitter: str=None
    state: State=None
    mergedName: str=None
    mergedOnlyName: str=None
    online_avatar: str=None
    avatar: str=None
    sponsor_logo: str=None

@dataclass_json
@dataclass
class Team:
    teamName: str
    losers: bool
    player: Dict[int, Player]
    score: int

@dataclass_json
@dataclass
class Ruleset:
    name: str
    neutralStages: List[str]
    counterpickStages: List[str]
    banByMaxGames: dict
    useDSR: bool
    useMDSR: bool
    banCount: int
    strikeOrder: List[str]
    videogame: str

@dataclass_json
@dataclass
class Score:
    team: Dict[int, Team]
    phase: str
    match: str
    best_of: int
    ruleset: Ruleset = None

@dataclass_json
@dataclass
class Commentary:
    team: str
    name: str
    real_name: str
    twitter: str
    pronoun: str

@dataclass_json
@dataclass
class Game:
    name: str
    smashgg_id: int=None
    codename: str=None

@dataclass_json
@dataclass
class TournamentInfo:
    eventName: str
    tournamentName: str
    numEntrants: int = None
    address: str = None


@dataclass_json
@dataclass
class TSHProgramState:
    game: Game
    tournamentInfo: TournamentInfo
    score: Score
    commentary: Dict[int, Commentary]=None

def _createPlayer(
        countryIso2:str,
        clubName:str,
        name:str
) :
    return Player(
        country=Country(asset=getNormalizedPath(os.path.join(TSH_COUNTRIES_FLAGS_BASE_PATH, f'{countryIso2}.png')), code=countryIso2),
        team = clubName,
        name=name
    )

def createTSHData(
        gameName:str,
        lhsName:str,
        lhsIsFromLosers:bool,
        lhsCountry:str,
        rhsName:str,
        rhsCountry:str,
        rhsIsFromLosers:bool,
        lhsSetScore:int,
        rhsSetScore:int,
        tournamentName:str,
        eventName:str,
        lhsClubName:str,
        rhsClubName:str,
        phaseName:str,
        bestOf:int,
        match:str):
    
    lhsPlayer = _createPlayer(lhsCountry, lhsClubName, lhsName)
    rhsPlayer = _createPlayer(rhsCountry, rhsClubName, rhsName)
    teams = {
        1: Team(lhsClubName, lhsIsFromLosers, {1: lhsPlayer}, lhsSetScore),
        2: Team(rhsClubName, rhsIsFromLosers, {1: rhsPlayer}, rhsSetScore)
    }
    return TSHProgramState(
        Game(gameName),
        TournamentInfo(eventName, tournamentName),
        Score(best_of=bestOf, match=match, phase=phaseName, team=teams)
    )

def writeTSHData(programState:TSHProgramState):
    with open(TSH_OUT_PROGRAM_STATE_PATH, 'w') as f:
        f.write(json.dumps(programState.to_dict(), indent=4))
    print(f'Wrote output to: {TSH_OUT_PROGRAM_STATE_PATH}')

'''
    END OF TSH PROGRAM STATE
'''


# we need to create a temporary web server to allow the index.html scoreboard to load!
# otherwise it'll faile with CORs errors.
path_to_serve = TSH_BASE_PATH  # Replace with the actual directory path

import http.server
import socketserver

PORT = 8080
DIRECTORY = TSH_BASE_PATH



# Define a request handler function
async def handle(request):
    return web.FileResponse(path_to_serve + request.path)

# Create an asynchronous web application
app = web.Application()

async def shutdown(request):
    print("will shutdown now")
    raise GracefulExit()

app.router.add_get("/{tail:.*}", handle)

# Set up the server
port = 8000  # Specify the port you want to use

runner = web.AppRunner(app)

# Start the server asynchronously
async def run_server():
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', port)
    await site.start()


async def stop_server():
    await runner.cleanup()

from aiohttp.web_runner import GracefulExit
  
async def html2png(html_file_path:str, output_file_path:str, swapP1andP2ScoreColors:bool=False):
    await run_server()
    print("after serving..")
    # Launch the browser
    browser = await launch(headless=True, defaultViewport={'width': 1920, 'height': 1080})
    # Create a new page
    page = await browser.newPage()

    # make background transparent
    await page._client.send('Emulation.setDefaultBackgroundColorOverride', {'color': {'r': 0, 'g': 0, 'b': 0, 'a': 0}})

    await page.goto(html_file_path)
    await page.waitForSelector("div.tournament_container")

    # have to wait for the animation from TSH to complete.
    await page.waitForFunction('document.querySelector("div.tournament_container").style.getPropertyValue("opacity") == 1')
     
    if(swapP1andP2ScoreColors):
        # have to wait for the animation from TSH to complete.
        val1 = await page.evaluate('getComputedStyle(document.querySelector(":root")).getPropertyValue("--p1-score-bg-color")')
        val2 = await page.evaluate('getComputedStyle(document.querySelector(":root")).getPropertyValue("--p2-score-bg-color")')
        await page.evaluate(f'document.querySelector(":root").style.setProperty("--p1-score-bg-color", "{val2}")')
        await page.evaluate(f'document.querySelector(":root").style.setProperty("--p2-score-bg-color", "{val1}")')

    print('waited and got the selector!')
  
    # Take a screenshot
    await page.screenshot({'path': output_file_path})

    # Close the browser
    await browser.close()

def getScoreboardFile(
        gameName:str,
        lhsName:str,
        lhsIsFromLosers:bool,
        lhsCountry:str,
        rhsName:str,
        rhsCountry:str,
        rhsIsFromLosers:bool,
        lhsSetScore:int,
        rhsSetScore:int,
        tournamentName:str,
        eventName:str,
        lhsClubName:str,
        rhsClubName:str,
        phaseName:str,
        bestOf:int,
        match:str,
        output_file_path:str,
        templateAssetFilePath:str=None,
        swapP1andP2ScoreColors:bool=False) -> str:

    """_summary_

    Args:
        lhsName (str): The name of the player on the LHS.
        rhsName (str): The name of the player on the RHS.
        lhsSetScore (int): LHS set score
        rhsSetScore (int): RHS set score.
        tournamentName (str): name of the tourney or event
        output_file_path (str): the file path to output to
        lhsClubName (str, optional): the LHS club name. Defaults to None.
        rhsClubName (str, optional): the RHS club name. Defaults to None.

    Returns:
        str: the file path to the output scoreboard PNG.
    """

    # write to TSH program state
    tshData = createTSHData(
        gameName,
        lhsName,
        lhsIsFromLosers,
        lhsCountry,
        rhsName,
        rhsCountry,
        rhsIsFromLosers,
        lhsSetScore,
        rhsSetScore,
        tournamentName,
        eventName,
        lhsClubName,
        rhsClubName,
        phaseName,
        bestOf,
        match)
    
    writeTSHData(tshData)
    # Now when we load the browser with pypeteer, it'll run as-if TSH is updating the data.
    # So no complex UX manipulation is required!
    
    if not templateAssetFilePath:
        templateAssetFilePath = TSH_LAYOUTS_MSBL_MSL_PATH
    fullPath = getNormalizedPath(templateAssetFilePath)
    link = f'http://localhost:{port}/{fullPath}';
    print(link)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(html2png(link, output_file_path, swapP1andP2ScoreColors))
    loop.run_until_complete(stop_server())

def getNormalizedPath(path:str, path_to_serve:str=path_to_serve):
    return os.path.normpath(os.path.relpath(os.path.abspath(path), path_to_serve)).replace("\\", "/")

def getScoreboardFilesForAllGames(
                         gameName:str,
                         eventName:str,
                         tournamentName:str,
                         bestOf:int,
                         phaseName:str,
                         startingLhsName:str,
                         startingLhsIsFromLosers:bool,
                         startingLhsTeamName:str,
                         startingRhsName:str,
                         startingRhsIsFromLosers:bool,
                         startingRhsTeamName:str,
                         startingLhsCountry:str,
                         startingRhsCountry:str,
                        winners:list[str],
                        outputFilePath:str,
                        lhsNames:list=None,
                        startingLhsSetScore:int=0,
                        startingRhsSetScore:int=0):
    inWinnersButNotCompeting = set(winners).difference(set([startingLhsName, startingRhsName]))
    if len(inWinnersButNotCompeting) > 0:
        raise ValueError(f'Exepcted that winners only contains either "{startingLhsName}" or "{startingRhsName}", but found {str(inWinnersButNotCompeting)}')
    if lhsNames and len(lhsNames) != len(winners):
        raise ValueError(f'Length mismatch: Expected lhsNames and winners to be the same length. OR for lhsNames to not be specified')
    elif not lhsNames:
        lhsNames = []
        names = [startingLhsName, startingRhsName]
        for i in range(len(winners)):
            lhsNames.append(names[i % 2])
    lhsNames.append(lhsNames[-1]) # for the final result.

    players = {
        startingLhsName : {
            'name': startingLhsName,
            'teamname': startingLhsTeamName,
            'setscore': startingLhsSetScore,
            'country': startingLhsCountry,
            'losers': startingLhsIsFromLosers
        },
        startingRhsName : {
            'name': startingRhsName,
            'teamname': startingRhsTeamName,
            'setscore': startingRhsSetScore,
            'country': startingRhsCountry,
            'losers': startingRhsIsFromLosers
        }
    }

    playerStates = []

    newStatePrevLhs = players[startingLhsName]
    newStatePrevRhs = players[startingRhsName]
    print(lhsNames)
    for i in range(len(lhsNames)):
        # xxhkxx, xxhkxx, xxhkxx
        # xxhkxx, michi, xxhkxx, xxhkxx
        # game1: 0-0
        # game2: 1-0
        # game3: 2-0
        # FINAL: 3-0
        state = {}
        if i > 0 and winners[i-1].lower() == newStatePrevRhs['name'].lower(): # rhs from last game won
            newStatePrevRhs['setscore'] += 1
        elif i > 0:
            newStatePrevLhs['setscore'] += 1
        if lhsNames[i] == newStatePrevLhs['name']:
            state['lhs'] = newStatePrevLhs
            state['rhs'] = newStatePrevRhs
        else:
            state['lhs'] = newStatePrevRhs
            state['rhs'] = newStatePrevLhs
        playerStates.append(state)
        newStatePrevLhs = copy.deepcopy(playerStates[-1]['lhs'])
        newStatePrevRhs = copy.deepcopy(playerStates[-1]['rhs'])        
        
    print(playerStates)

    # first create all the relevant scoreboards
    outputPath = f'out/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    Path(outputFilePath).mkdir(parents=True, exist_ok=True)
    scoreboards = []
    lhsSetScore=startingLhsSetScore
    rhsSetScore=startingRhsSetScore
    gameNo = 0
    game1LhsName = lhsNames[0]
    swapped = []
    for playerState in playerStates:
        print('\tGetting scoreboard for gameNo=',gameNo)
        scoreboardName = os.path.join(outputFilePath, f'scoreboard_game_{gameNo}.png')
        gameNo = playerState['lhs']['setscore'] + playerState['rhs']['setscore'] + 1
        match = f'Game {gameNo}'
        if gameNo > (int(bestOf)//2)+1:
            match = 'FINAL RESULT'
        getScoreboardFile(
                gameName=gameName,
                lhsName=playerState['lhs']['name'],
                lhsIsFromLosers=playerState['lhs']['losers'],
                lhsCountry=playerState['lhs']['country'],
                rhsName=playerState['rhs']['name'],
                rhsCountry=playerState['rhs']['country'],
                rhsIsFromLosers=playerState['rhs']['losers'],
                lhsSetScore=playerState['lhs']['setscore'],
                rhsSetScore=playerState['rhs']['setscore'],
                tournamentName=tournamentName,
                eventName=eventName,
                lhsClubName=playerState['lhs']['teamname'],
                rhsClubName=playerState['rhs']['teamname'],
                phaseName=phaseName,
                bestOf=bestOf,
                match=match,
                output_file_path=scoreboardName,
                swapP1andP2ScoreColors=game1LhsName!=playerState['lhs']['name']
        )
        scoreboards.append(scoreboardName)
        prevLhsName = playerState['lhs']['name']

    return scoreboards




# getScoreboardFile(
#         gameName='MSBL',
#         lhsName='XXhkXX',
#         lhsIsFromLosers=False,
#         lhsCountry='us',
#         rhsName='Coolade',
#         rhsCountry='us',
#         rhsIsFromLosers=False,
#         lhsSetScore=0,
#         rhsSetScore=0,
#         tournamentName='TOURNAMENT',
#         eventName='FUNNY SPLIT',
#         lhsClubName='EVO',
#         rhsClubName='LOUD',
#         phaseName='Playoffs',
#         bestOf=7,
#         match='ROUND 3',
#         output_file_path='screnshot.png',
#         swapP1andP2ScoreColors=True
# )