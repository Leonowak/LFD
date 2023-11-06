#!/bin/bash
#SBATCH --job-name=LFDfin_distil_emotion_EX
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --output=LFDfin_distil_emotion_EX_test3.log
#SBATCH --mem=64GB

module purge
module load Python/3.9.6-GCCcore-11.2.0

source $HOME/venvs/first_env/bin/activate

python3 --version
which python3

python3 LFDfin_distilbert_EX.py -trE "train_offenseval_hatexplain.json" -dE "dev_offenseval_hatexplain.json" -lr "0.0001" -bs "128" -e "1" -sl "100"

deactivate