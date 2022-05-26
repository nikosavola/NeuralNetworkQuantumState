#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --job-name=nnqs_phase_sweep
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2GB
#SBATCH --array=1-4
#SBATCH --output=nnqs_phase_sweep_%a.out

module load anaconda gcc openmpi

# install necessary packages
pip install --upgrade "jax[cpu]" "netket[mpi]" typing-extensions

i=$SLURM_ARRAY_TASK_ID
number_of_J2=12

# Split to (total) array jobs that have n/i jobs each
for n in $(seq 1 $number_of_J2); do
    if [ $((n % i)) -eq 0 ]; then
        echo "$n"
        srun python run_phase_sweep.py --hyperparams "data/hyperparams.json" --J2_idx "$n" --J2_max "$number_of_J2"
    fi
done