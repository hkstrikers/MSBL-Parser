#!/usr/bin/env python
import argparse
from datetime import date, datetime, time
import glob
from dataclasses import dataclass
import os
from pathlib import Path
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip, ImageClip
from htmlToPng import getScoreboardFilesForAllGames
import subprocess

# Custom function to validate multiple file extensions
def readable_file(value):
    try:
        with open(value, 'r'):
            pass
    except IOError:
        raise argparse.ArgumentTypeError(f"{value} is not a readable file")
    return value

def getDuration(time):
    return (datetime.combine(date.min, gameStartTimes[i+1]) - datetime.min).total_seconds()

def parse_time(time_str):
    try:
        minutes, seconds = map(int, time_str.split(":"))
        if 0 <= minutes <= 59 and 0 <= seconds <= 59:
            return minutes*60+seconds
        else:
            raise ValueError("Invalid minutes or seconds")
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid time format. Use 'HH:MM' (0-23 hours, 0-59 minutes)")


parser = argparse.ArgumentParser(description="A command-line tool to edit together MSL League videos.")
# Create a mutually exclusive group for the two arguments
parser.add_argument('--input_files', '-i', required=True, nargs='+', type=str, help='List of readable files; the output video will concatenate these files in alphabetical order.')
parser.add_argument('--intro_file', '-b', type=str, help='The intro clip to attach to the beginning of the video')
parser.add_argument('--intro_trim_first_x_seconds', type=int, required=False, help='Trim the intro to just the first X seconds')
parser.add_argument('--outro_file', '-e', type=str, help='The outro clip to attach to the beginning of the video')
parser.add_argument('--output', '-o', required=True, type=str, help='Path to the output file')
parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose mode')
parser.add_argument('--codec', '-c', default="hevc_nvenc", type=str, help='The codec to output the vidoe with')
parser.add_argument('--threads', '-t', default=None, type=int, help='The codec to output the vidoe with')
parser.add_argument('--overlayFile', '-l', required=False, type=str, help='The overlay PNG to use')
parser.add_argument('--gameName', required=True, type=str, help='The name of the game played')
parser.add_argument('--eventName', required=True, type=str, help='The name of the start gg event')
parser.add_argument('--tournamentName', required=True, type=str, help='The name of the tournament')
parser.add_argument('--bestOf', required=True, type=str, help='The best of # of the series (like best of 7)')
parser.add_argument('--phaseName', required=True, type=str, help='The name of the phase of the tournament (like Winners Semis)')
parser.add_argument('--startingLhsName', required=True, type=str, help='The name of the player who started on the LHS in the first game')
parser.add_argument('--startingLhsIsFromLosers', default=False, required=False, type=str, help='Whether the player starting on the LHS is from losers or not')
parser.add_argument('--startingLhsCountry', required=True, type=str, help='The ISO2 country code for the player starting on the LHS')
parser.add_argument('--startingLhsTeamName', required=True, type=str, help='The team name for the player starting on the LHS (like EVO)')
parser.add_argument('--startingRhsName', required=True, type=str, help='The name of the player who started on the RHS in the first game')
parser.add_argument('--startingRhsIsFromLosers', default=False, required=False, type=str, help='Whether the player starting on the RHS is from losers or not')
parser.add_argument('--startingRhsCountry', required=True, type=str, help='The ISO2 country code for the player starting on the RHS')
parser.add_argument('--startingRhsTeamName', required=True, type=str, help='The team name for the player starting on the RHS (like EVO)')
parser.add_argument('--winners', required=False, type=str, nargs='+', help='The names of the winners in order of each game. Otherwise parses which side won from the video.')
parser.add_argument('--gameStartTimes', required=False, type=parse_time, nargs='+', help='The times that each game ends in the video + a final time indicating when to STOP displaying the scoreboard.')
parser.add_argument('--stopScoreboardDisplayTime', required=False, type=parse_time, help='The time to stop the scoreboard display.')
parser.add_argument('--startScoreboardDisplayTime', required=False, type=parse_time, help='The time to start the scoreboard display. To avoid displaying it during lag tests and such.')
parser.add_argument('--lhsNames', required=False, type=str, nargs='+', help='The list of player names playing on the LHS each game.')
parser.add_argument('--startingLhsSetScore', default=0 , required=False, type=int, help='The starting set score of the player who started on the LHS')
parser.add_argument('--startingRhsSetScore', default=0 , required=False, type=int, help='The starting set score of the player who started on the RHS')
parser.add_argument('--tempOutputDirectory', default='out', required=False, type=str, help='The base directory to store temporary output.')
parser.add_argument('--ffmpegPath', default=None, required=False, type=str, help='The path to ffmpeg, otherwise just tries to run it in system.')

args = parser.parse_args()
print(f"arguments are: {args}")

if args.gameStartTimes and args.winners and len(args.gameStartTimes) != len(args.winners): 
    raise ValueError('Need # games time for gameStartTimes. Last time should represent when to stop the scoreboard display')

file_paths = []
for pattern in args.input_files:
    expanded_paths = glob.glob(pattern)
    file_paths.extend(expanded_paths)

sorted_file_paths = sorted(file_paths)

print("Reading input files from:", sorted_file_paths)

input_videos = [VideoFileClip(fp) for fp in sorted_file_paths]
print(f'loaded {len(input_videos)} videos')
target_resolution = input_videos[0].size
input_videos[0] = input_videos[0].crossfadein(1)
total_duration = sum([input_video.duration for input_video in input_videos])
gameStartTimes = args.gameStartTimes

intro_clip = None
intro_duration = args.intro_trim_first_x_seconds
if args.intro_file:
    intro_clip = VideoFileClip(args.intro_file)
    if not intro_duration:
        intro_duration = intro_clip.duration
    total_duration += intro_duration
    for i in range(len(gameStartTimes)):
        gameStartTimes[i] += intro_duration

outro_clip = None
if args.outro_file:
    outro_clip = VideoFileClip(args.outro_file)



baseOutputPath = args.tempOutputDirectory
outputPath = os.path.join(baseOutputPath, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
videosPath = os.path.join(outputPath, 'videos')
Path(videosPath).mkdir(parents=True, exist_ok=True)

input_video_path = sorted_file_paths[0]

# if len(input_videos) > 1:
#     input_video_path = os.path.join(videosPath, 'tmpMergedVideo.mp4')
#     print('Writing temp video - merged videos.')
#     input_video.write_videofile(input_video_path)

scoreboards = getScoreboardFilesForAllGames(
                         gameName=args.gameName,
                         eventName=args.eventName,
                         tournamentName=args.eventName,
                         bestOf=args.bestOf,
                         phaseName=args.phaseName,
                         startingLhsName=args.startingLhsName,
                         startingLhsIsFromLosers=args.startingLhsIsFromLosers,
                         startingLhsTeamName=args.startingLhsTeamName,
                         startingRhsName=args.startingRhsName,
                         startingRhsIsFromLosers=args.startingRhsIsFromLosers,
                         startingRhsTeamName=args.startingRhsTeamName,
                         startingLhsCountry=args.startingLhsCountry,
                         startingRhsCountry=args.startingRhsCountry,
                        winners=args.winners,
                        lhsNames=args.lhsNames,
                        startingLhsSetScore=args.startingLhsSetScore,
                        startingRhsSetScore=args.startingRhsSetScore,
                        outputFilePath=os.path.join(outputPath, 'scoreboards'))

print(gameStartTimes)
gameStartTimes.insert(0,0 if not intro_clip else intro_clip.duration)
stopDisplayTime = args.stopScoreboardDisplayTime if args.stopScoreboardDisplayTime else total_duration
updated_scoreboards = []
@dataclass
class ImageInfo:
    imageFile:str
    startTimeInSeconds:int
    endTimeInSeconds:int

print(total_duration)
for i in range(len(scoreboards)):
    endTime =min(total_duration, stopDisplayTime)
    if i+1 < len(gameStartTimes):
        endTime = gameStartTimes[i+1] 
    startTime = gameStartTimes[i]
    scoreboard = scoreboards[i]
    imageInfo = ImageInfo(scoreboard, startTime, endTime)
    print(str(imageInfo))
    updated_scoreboards.append(imageInfo)

width, height = VideoFileClip(sorted_file_paths[0]).size

def runFfmpegOverlay(videoPaths:str, overlays:list[ImageInfo], outputFile:str, width:int, height:int, introVideoPath:str=None, intro_duration_seconds:int=None, introHasAudioTrack:bool=False, outroVideoPath:str=None, outroHasAudioTrack:bool=False):
    # alright so this is just terrible and overly complicated. Basically moviepy was too slow -- so instead I'm calling ffmpeg directly to do the video composition.
    ffmpegPath = args.ffmpegPath if args.ffmpegPath else 'ffmpeg'
    commandStr = ffmpegPath
    # add all video inputs
    commandStr += ' ' + ' '.join([f'-i "{v}"' for v in videoPaths])
    # add all image inputs
    commandStr += ' ' + ' '.join([f'-i "{i.imageFile}"' for i in overlays])
    nullSrcArg = len(videoPaths) + len(overlays)
    commandStr += ' -f lavfi -i anullsrc'
    introVideoArg = len(videoPaths) + len(overlays) + 1
    introVideoCount = 1 if introVideoPath is not None else 0
    if introVideoPath:
        commandStr += f' -i {introVideoPath}'
    outroVideoArg = len(videoPaths) + len(overlays) + 2
    outroVideoCount = 1 if outroVideoPath is not None else 0    
    if outroVideoPath:
        commandStr += f' -i {outroVideoPath}'    
    # add filters
    commandStr += ' -filter_complex "'
    currentVideoOutput = 1
    introAudioTrackRef = '[ai]'
    if introVideoPath:
        commandStr += f'[{introVideoArg}:v]scale=1920:1080' + ('' if not intro_duration_seconds else f',trim=duration={int(intro_duration_seconds)}') + '[vi];'
        if not introHasAudioTrack:
            commandStr += f'[{nullSrcArg}:a]atrim=duration={int(intro_duration_seconds)}{introAudioTrackRef};'
    outroAudioTrackRef = '[ao]'
    if outroVideoPath:
        commandStr += f'[{outroVideoArg}:v]scale=1920:1080[vo];'
        if not outroHasAudioTrack:
            commandStr += f'[{nullSrcArg}:a]atrim=duration={int(intro_duration_seconds)}{outroAudioTrackRef};'        
    if introVideoPath:
        commandStr += f'[vi] ' + (f'[{introVideoArg}:a]' if introHasAudioTrack else introAudioTrackRef)
    for i in range(len(videoPaths)):
        commandStr += f'[{i}:v] [{i}:a] '
    if outroVideoPath:
        commandStr += f'[vo] ' + (f'[{outroVideoArg}:a]' if outroHasAudioTrack else outroAudioTrackRef)
    commandStr += f'concat=n={len(videoPaths) + introVideoCount + outroVideoCount}:v=1:a=1[v{currentVideoOutput}][a]'
    for i in range(len(overlays)):
        overlayArg = len(videoPaths) + i
        st = overlays[i].startTimeInSeconds
        et = overlays[i].endTimeInSeconds
        timeArg= f'between(t,{st},{et})' if et is not None else f'gt(t,{st})'
        commandStr += f";[v{currentVideoOutput}][{overlayArg}]overlay=x=0:y=0:enable='{timeArg}'[v{currentVideoOutput+1}]"
        currentVideoOutput+=1
    commandStr+= f'" -vsync 0 -map "[v{currentVideoOutput}]" -map "[a]" {outputFile}'
    print(f'running command:\n\n{commandStr}')
    os.system(commandStr)

#mergedWithOverlaysFile = os.path.join(videosPath, 'mergedWithOverlays.mp4')
Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)
runFfmpegOverlay(sorted_file_paths, updated_scoreboards, args.output, width, height, introVideoPath=args.intro_file, intro_duration_seconds=intro_duration, introHasAudioTrack=(intro_clip is not None) and (intro_clip.audio is not None), outroVideoPath=args.outro_file, outroHasAudioTrack=(outro_clip is not None) and (outro_clip.audio is not None))
print(f'Output the file to {args.output}')

# [v1][2]overlay=x=0:y=0:enable='between(t,44,61)'[v2]; 
# [v2][3]overlay=x=0:y=0:enable='gt(t,112)'[v3]"
 
 #ffmpeg -i "C:\Users\hkuser\Videos\xxhkxxvsmichi.mkv" -i "C:\wd\MSBL-Parser\src\videoutils\out\2023-08-26_14-46-28\scoreboard_game_0.png"
#  -i "C:\wd\MSBL-Parser\src\videoutils\out\2023-08-26_14-46-28\scoreboard_game_1.png"
#  -i "C:\wd\MSBL-Parser\src\videoutils\out\2023-08-26_14-46-28\scoreboard_game_2.png"
#  -i "C:\wd\MSBL-Parser\src\videoutils\out\2023-08-26_14-46-28\scoreboard_game_3.png" 
# -filter_complex 
# "[0][1]overlay=x=0:y=0:enable='between(t,23,27)'[v1]; 
# [v1][2]overlay=x=0:y=0:enable='between(t,44,61)'[v2]; 
# [v2][3]overlay=x=0:y=0:enable='gt(t,112)'[v3]" -map "[v3]" -map 0:a out.mp4

    
# def overlayImagesOnVideoUsingFfmpeg(videoFile:str, imagesFiles:ImageInfo, outputFileName:string):

# if intro_clip:
#     concatenate_clips.insert(0, intro_clip.resize(target_resolution))
# if outro_clip:
#     concatenate_clips.append(outro_clip.resize(target_resolution))

# concatenated_clip = concatenate_videoclips(concatenate_clips)

# concatenated_clip.write_videofile(args.output, threads=args.threads, codec=args.codec)