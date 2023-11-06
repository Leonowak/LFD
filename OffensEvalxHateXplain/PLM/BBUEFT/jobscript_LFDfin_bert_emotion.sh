#!/bin/bash
#SBATCH --job-name=LFDfin_bert_emotion
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --output=LFDfin_bert_emotion_test3.log
#SBATCH --mem=64GB

module purge
module load Python/3.9.6-GCCcore-11.2.0

source $HOME/venvs/first_env/bin/activate

python3 --version
which python3

python3 LFDfin_bert_base_uncased_emotion.py -tr "train_offens.json" -d "dev_offens.json" -lr "0.0001" -bs "64" -e "2" -sl "150"

deactivate