#!/bin/sh
#SBATCH --job-name spectrum_kernel_computation
#SBATCH --error spkc.%j.err
#SBATCH --output spkc.%j.out
#SBATCH --mail-user stefano.campese@phd.unipd.it
#SBATCH --mail-type END,FAIL
#SBATCH --partition allgroups
#SBATCH --ntasks 1
#SBATCH --cpu-per-task 32
#SBATCH --mem 128G
#SBATCH --time 24:00:00
#SBATCH –-gres=gpu:1
cd $SLURM_SUBMIT_DIR
srun singularity exec --nv container.sif python /opt/workspace/script.py
