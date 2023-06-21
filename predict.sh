#!/bin/bash

#SBATCH --job-name=AI-HERO_energy_baseline_prediction
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --output=slurm_log/predict_%j.out

export CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=76

data_workspace=/hkfs/work/workspace/scratch/ih5525-E4/AI-HERO-2-Energy/energy-train-data/
group_workspace=/hkfs/work/workspace/scratch/ih5525-E4/AI-HERO-2-Energy

module load compiler/gnu/11
module load mpi/openmpi/4.0
module load lib/hdf5/1.12
module load devel/cuda/11.8

source ${group_workspace}/production_env/bin/activate
torchrun --nproc-per-node=4 ${group_workspace}/predict.py ${data_workspace} -m ${group_workspace}/models/fast-silence-187/checkpoint_best.pt --batch=4 --backbone=resnet18 --grayscale=True 
# srun python ${group_workspace}/predict.py ${data_workspace} -m ${group_workspace}/models/rose-bush-138/checkpoint_best.pt --batch=2 --backbone=resnet18 --grayscale=True --normalize=True

