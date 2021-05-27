#!/bin/sh
#SBATCH --job-name spectrum_kernel_computation
#SBATCH --error spkc.%j.err
#SBATCH --output spkc.%j.out
#SBATCH --mail-user stefano.campese@phd.unipd.it
#SBATCH --mail-type END,FAIL
#SBATCH --partition allgroups
#SBATCH --ntasks 16
#SBATCH --mem 32G
#SBATCH --time 02:25:00
#SBATCH –-gres=gpu:1
cd $SLURM_SUBMIT_DIR
srun singularity exec container.sif python script.py
