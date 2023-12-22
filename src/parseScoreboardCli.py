
import argparse
import sys
sys.path.append(".") # almost every `.py` need this

from parsing import tryParseImageUsingDefaultSettings
from PIL import Image
from io import BytesIO
import base64
import json
from constants import GamePhase, GameSide, KeyItem
parser = argparse.ArgumentParser(description="A command-line tool to process MSBL scoreboards in Python.")
# Create a mutually exclusive group for the two arguments
parser.add_argument('--input', '-i', required=False, type=str, help='The input to be passed to the script')

parser.add_argument("-v", "--verbose", required=False,  help="increase output verbosity",
                    action="store_true")
parser.add_argument("-p", "--prettyprint", required=False,  help="pretty print the script output - useful for debugging should be off otherwise",
                    action="store_true")


args = parser.parse_args()
debug = False
if args.verbose:
    debug=True
prettyPrint = False
if args.prettyprint:
    prettyPrint=True

imInput = None
try:
    imInput = input()
except EOFError:
     raise Exception("no data provided to input function")

im = Image.open(BytesIO(base64.b64decode(imInput)))
im = im.resize((1920,1080))

parsed, _ = tryParseImageUsingDefaultSettings(im, GamePhase.END_GAME_SCOREBOARD, useConfigAsBackup=True, debug=debug)
if parsed.parsingResults[KeyItem.SCOREBOARD_PASSES_CHECK][GameSide.NONE].parsedValue != "PASSES":
    print(json.dumps({'error': f"Could not parse this image - please make sure it's a screenshot of the end game scoreboard from the Switch console. Parsed {parsed.parsingResults[KeyItem.SCOREBOARD_PASSES_CHECK][GameSide.NONE].parsedValue}"}))
else:
    result = {k: {k2: v2.to_dict() for k2,v2 in v.items()} for k,v in parsed.parsingResults.items()}
    #typesof = {k: {k2: {k3: str(type(v3)) for k3,v3 in v2.to_dict().items()} for k2,v2 in v.items()} for k,v in parsed.parsingResults.items()}
    if prettyPrint:
        print(json.dumps(result, indent=4), flush=True)
    else:
        print(json.dumps(result), flush=True)
