#!/bin/bash
#SBATCH --job-name=RF_baseline_OE_test
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --output=RF_baseline_OE_test.log
#SBATCH --mem=64GB

module purge
module load Python/3.9.6-GCCcore-11.2.0

source $HOME/venvs/my_env/bin/activate

python3 --version
which python3

python bag-of-words.py -tf dev_offens.json -df test_offens.json -t --algorithm rf 


deactivate
