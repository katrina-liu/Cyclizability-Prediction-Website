#!/bin/bash
#SBATCH -c 2
#SBATCH -t 0-01:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -o pred_cyc_%j.out
#SBATCH --mail-user= ##### FILL IN #####
#SBATCH --mail-type=FAIL
#SBATCH --mem=20G
 
module load gcc/9.2.0
module load cuda/12.1

source activate cyc_pred

python read_fasta.py $1 $2
