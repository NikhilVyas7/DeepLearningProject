#!/bin/bash
#SBATCH -N 1                # Request 1 node
#SBATCH -t 8:00:00          # Set time limit to 6 hours
#SBATCH --gres=gpu:2        # Request 2 GPU, if I want more, got to switch to DataDistributedParallel
#SBATCH --cpus-per-task=20  # Request 15 CPUs per task
#SBATCH --mem=200G          #Request 200 GB of memory for cacheing the whole dataset
# Load any necessary modules
module load anaconda3
conda info --envs #For some reason this makes it work, wtf

# Run your command or script

conda run -n dl_project python diffusion_train.py