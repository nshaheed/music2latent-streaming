#!/bin/bash
#
#SBATCH --job-name=music2lat
#
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=24G

ml load python/3.9.0
ml load libsndfile
ml load gcc/14.2.0

nvidia-smi

uv run launch.py



