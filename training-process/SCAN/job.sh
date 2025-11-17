#!/bin/bash
#SBATCH --account=pintxo
#SBATCH --partition=pintxo
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --ntasks-per-node=1         
#SBATCH --nodes=1         
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=12G         # memory per cpu-core (4G is default)
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --job-name="water"
#SBATCH --gres=gpu:1        # number of gpus per node

pwd; hostname; date

module load Anaconda3

source /scratch/ppena/hybrid-ab-initio-empirical/.venv/bin/activate

python train.py

date
