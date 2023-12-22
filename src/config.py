import json
import os
from helpers import ParsingConfig

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as f:
    CONFIG = json.load(f)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parseConfig.json'), 'r') as f:
    PARSE_CONFIG = ParsingConfig.from_json(f.read())