#!/bin/bash
#SBATCH --job-name=lab5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=lab5-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
srun .venv/bin/python -u lab5.py

