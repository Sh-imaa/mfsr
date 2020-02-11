#!/bin/bash
#SBATCH --account=rpp-bengioy            # Yoshua pays for your job
#SBATCH --cpus-per-task=6                # Ask for 6 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --time=10:00:00                   # The job will run for 13 hours
#SBATCH -o /scratch/shimaa/logs/mfsr-%j.out  # Write the log in $SCRATCH
#SBATCH -e /scratch/shimaa/logs/mfsr-%j.err  # Write the err in $SCRATCH

# change the log directories based on your space
# TODO: make the logging directories map to current user space

# Copy and unzip the raw data to the compute node
# this assumes you have the raw_data on your scratch space,
# if it doesn't, you can copy it from /scratch/shimaa (everyone has read access)
cp $SCRATCH/data/probav_data.zip $SLURM_TMPDIR
unzip $SLURM_TMPDIR/probav_data.zip -d $SLURM_TMPDIR

# singularity 3.5 path issue
REVERSED_PATH="$(tr ':' '\n'  <<< $PATH | tac | paste -s -d ':')"
export PATH="$REVERSED_PATH"

module load singularity/3.5
module load cuda/9.2
cd $HOME/mfsr
singularity exec --nv --bind $SLURM_TMPDIR,$SCRATCH /scratch/shimaa/images/mfsr_0_1.sif python3 -m src.train --config cluster/c_config-replace.json

