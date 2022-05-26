#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --job-name=nnqs_training
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=3GB
#SBATCH --output=nnqs_training.out

module load anaconda gcc openmpi

# install necessary packages
pip install --upgrade "jax[cpu]" "ray[tune]" "netket[mpi]" hyperopt hiplot typing-extensions

srun python run_tuning.py --num_samples 300