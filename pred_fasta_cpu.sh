#!/bin/bash
#SBATCH -c 2                               # 2 core
#SBATCH -t 4-00:00                         # Runtime of 4 days, in D-HH:MM format
#SBATCH -p medium                           # Run in medium partition
#SBATCH -o slurm_out/pred_cyc_%j.out
#SBATCH --mail-user= ##### FILL IN #####
#SBATCH --mail-type=FAIL
#SBATCH --mem=5G

source activate cyc_pred

python read_fasta.py $1 $2