#!/bin/sh

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 INPUT-GIF OUTPUT-MP4 FRAMERATE"
    exit 1
fi

ffmpeg -r "$3" -i "$1" -vcodec libx264 -pix_fmt yuv420p "$2"
