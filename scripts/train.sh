
#!/bin/bash
#SBATCH -N 1                # Request 1 node
#SBATCH -t 8:00:00          # Set time limit to 6 hours
#SBATCH --gres=gpu:2        # Request 1 GPU
#SBATCH --cpus-per-task=20  # Request 15 CPUs per task

# Load any necessary modules
module load anaconda3
conda activate dl_project
# Run your command or script
python train.py