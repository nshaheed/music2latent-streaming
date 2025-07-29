#!/bin/bash
#
#SBATCH --job-name=conversion
#
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

# Set the base directory to the current directory or use the first argument
# BASE_DIR="${1:-.}"
BASE_DIR="/scratch/users/nshaheed/mtg-jamendo/"

echo $BASE_DIR

ml load system
ml load ffmpeg

# Find all .mp3 files and convert each one
# find "$BASE_DIR" -type f -iname "*.mp3" | while read -r mp3_file; do
for mp3_file in `find ${BASE_DIR} -name "*.mp3" -type f`; do
    # Create corresponding .wav filename
    wav_file="${mp3_file%.mp3}.wav"

    # Skip conversion if .wav file already exists
    if [ -f "${wav_file}" ]; then
        echo "Skipping existing: $wav_file"
        continue
    fi

    echo "Converting: $mp3_file → $wav_file"
    ffmpeg -y -i "${mp3_file}" "${wav_file}"
done
