#!/bin/bash
#SBATCH --job-name=dataset-download   # Job name
#SBATCH --output=download-%j.out      # Output file (%j will be replaced by the job ID)
#SBATCH --error=download-%j.err       # Error file (%j will be replaced by the job ID)
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --time=10:00:00               # Set the max time limit (hh:mm:ss)
#SBATCH --mem=8G                      # Memory allocation (adjust as needed)
#SBATCH --cpus-per-task=1             # Number of CPU cores (adjust as needed)

module load anaconda3

conda activate dl_project

cd /home/hice1/$USER/scratch/DeepLearningProject


# Sync the S3 bucket with your local directory
aws s3 sync s3://radiantearth/landcovernet/ ./data --endpoint-url=https://data.source.coop
