#!/bin/bash
#SBATCH --job-name=RF_features_HX_test
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --output=RF_features_HX_test.log
#SBATCH --mem=64GB

module purge
module load Python/3.9.6-GCCcore-11.2.0

source $HOME/venvs/my_env/bin/activate

python3 --version
which python3

python bag-of-words.py -tf dev_offenseval_hatexplain.json -df test_offenseval_hatexplain.json -t --algorithm rf --n_estimators 300 --min_samples_split 2 -ngram 7 -char 1 -pos


deactivate
