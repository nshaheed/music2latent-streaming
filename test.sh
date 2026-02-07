#!/usr/bin/bash
#SBATCH --job-name=train_test
#SBATCH --output=train_test.%j.out
#SBATCH --error=train_test.%j.err
#SBATCH --time=0:20:00
#SBATCH -p hns,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=24G

# the magic configuration of software versions that gets this to work

ml load system ruse
ml load uv
ml load gcc/12.4.0
ml load go/1.18.2
ml load libsndfile

# torchhub model downloads fail otherwise
# https://stackoverflow.com/questions/77442172/ssl-certificate-verify-failed-certificate-verify-failed-unable-to-get-local-is
export SSL_CERT_FILE=$(uv run python -m certifi)

uv run launch.py --config configs/config_dev.py
# uv run launch.py --config configs/config_env_relu.py
# cd music2latent
# uv run fad_run.py
