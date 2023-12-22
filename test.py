import base64
import subprocess

# Read the Base64-encoded string from the file
file_path = "../RoboticNightmare/tmp_base64.txt"

try:
    with open(file_path, "r") as file:
        base64_string = file.read().strip()
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    exit(1)

# Decode the Base64 string
try:
    decoded_data = base64_string
except Exception as e:
    print(f"Error decoding Base64 string: {e}")
    exit(1)

# Pass the decoded string as input to the "parseScoreboardCli" script
script_path = "src/parseScoreboardCli.py"

try:
    result = subprocess.run(["python", script_path, "--verbose"], input=decoded_data, text=True, capture_output=True)
    print("Output from parseScoreboardCli:")
    print(result.stdout)
    if result.stderr:
        print("Error:", result.stderr)
except FileNotFoundError:
    print(f"Error: Script '{script_path}' not found.")
    exit(1)
except Exception as e:
    print(f"Error running script: {e}")
    exit(1)
