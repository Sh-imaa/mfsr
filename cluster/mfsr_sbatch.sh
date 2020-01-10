#!/bin/bash
#SBATCH --account=rpp-bengioy            # Yoshua pays for your job
#SBATCH --cpus-per-task=6                # Ask for 6 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --time=1:00:00                   # The job will run for 13 hours
#SBATCH -o /scratch/shimaa/logs/mfsr-%j.out  # Write the log in $SCRATCH
#SBATCH -e /scratch/shimaa/logs/mfsr-%j.err  # Write the err in $SCRATCH

# change the log directories based on your space
# TODO: make the logging directories map to current user space

# Copy and unzip the raw data to the compute node
# this assumes you have the raw_data on your scratch space,
# if it doesn't, you can copy it from /scratch/shimaa (everyone has read access)
cp $SCRATCH/data/probav_data.zip $SLURM_TMPDIR
unzip $SLURM_TMPDIR/raw_glaciers_data.zip -d $SLURM_TMPDIR

module load singularity/3.4
cd $HOME/mfsr/src
singularity exec --bind $SLURM_TMPDIR /scratch/shimaa/images/mfsr.sif python3 train.py --config ../cluster/c_config.json

