#!/bin/bash

#SBATCH --mail-user=vkapoor@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -J trainSTAVim_batch_size2
#SBATCH --output=/home/vkapoor/logs/dl/trainSTAVim_%j.out
#SBATCH --error=/home/vkapoor/logs/dl/trainSTAVim_%j.err

#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=16G

#SBATCH --gres=gpu:1
#SBATCH -C A100|H100

#SBATCH -p short
#SBATCH -t 23:00:00

module load cuda
conda activate dl
python train.py --output_dir /home/vkapoor/workspace/Deep_Learning_Fall24/weights/3/ --gpu_id 0 --root_path /home/vkapoor/workspace/Deep_Learning_Fall24/final_project/STA-UNet/data/Synapse/ --n_gpu 1 --batch_size 2
