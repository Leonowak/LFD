#!/bin/bash
#SBATCH --job-name=LFD3_distilbert-base-uncased-finetuned
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --output=LFD3_distilbert-base-uncased-finetuned.log
#SBATCH --mem=64GB

module purge
module load Python/3.9.6-GCCcore-11.2.0

source $HOME/venvs/first_env/bin/activate

python3 --version
which python3

python3 distilbert-base-uncased-fine-tuned.py -tr "train.txt" -d "dev.txt"

deactivate