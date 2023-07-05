#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10

#echo $(scontrol show hostnames $SLURM_JOB_NODELIST)
#source ~/.bashrc
#conda activate graph-aug

#echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES
#
#echo "python main.py --configs $config --num_workers 8 --devices $CUDA_VISIBLE_DEVICES"

if [ -z "$1" ]; then
  echo "empty cuda input!"
  cuda=0
else
  cuda=$1
fi

dataset=DD

# TU datasets
for batch_size in 128; do
  for hidden in 128; do
    CUDA_VISIBLE_DEVICES=$cuda python train_TUs.py --dataset $dataset --batch-size $batch_size --dim-hidden $hidden --not_extract_node_feature
  done
done

dataset=ENZYMES

# TU datasets
for batch_size in 128; do
  for hidden in 128; do
    CUDA_VISIBLE_DEVICES=$cuda python train_TUs.py --dataset $dataset --batch-size $batch_size --dim-hidden $hidden --not_extract_node_feature
  done
done
