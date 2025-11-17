#!/bin/bash
#SBATCH --qos=long
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --ntasks-per-node=1         
#SBATCH --nodes=1         
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=24G         # memory per cpu-core (4G is default)
#SBATCH --time=2-00:00:00          # total run time limit (HH:MM:SS)
#SBATCH --job-name="gdr500scan"  # name of the job
#SBATCH --gres=gpu:1        # number of gpus per node
#SBATCH --constraint=a100-pcie

pwd; hostname; date

source /scratch/ppena/hybrid-ab-initio-empirical/.venv/bin/activate

# Show GPU
echo "GPU:"
nvidia-smi
echo "Cuda ID: $CUDA_VISIBLE_DEVICES"

python train.py

date
