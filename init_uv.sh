#!/usr/bin/bash
#SBATCH --job-name=train_test
#SBATCH --output=train_test.%j.out
#SBATCH --error=train_test.%j.err
#SBATCH --time=1:00:00
#SBATCH -p hns,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=24G

# the magic configuration of software versions that gets this to work

ml load system ruse
ml load uv
ml load gcc/12.4.0
ml load go/1.18.2
ml load libsndfile

ruse uv run launch.py --config configs/config.py
