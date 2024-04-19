#!/bin/bash -l

#$ -P caad       # Specify the SCC project name you want to use
#$ -l h_rt=03:00:00   # Specify the hard time limit for the job
#$ -N GNN           # Give job a name
#$ -l gpus=1
#$ -l gpu_c=6.0

module load python3

module load cuda/11.8

python3 trainAE.py
