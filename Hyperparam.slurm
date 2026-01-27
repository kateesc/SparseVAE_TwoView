#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --mem=16G
#SBATCH --output=output_%A_%a.txt
#SBATCH --error=error_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-125

module load mamba
source activate conda-example

experiment_note="mocsMM-HyperparamTuning-EarlyAll"
disease="plant"
#disease="kirc"
lr=(0.0001 0.0003 0.0005 0.0007 0.0009)
batch_size=(32 64 128 256 512)
wd=(0.0001 0.0003 0.0005 0.0007 0.0009)

total_lr=${#lr[@]}
total_wd=${#wd[@]}
total_batch_size=${#batch_size[@]}
task_id=$((SLURM_ARRAY_TASK_ID-1))

lr_index=$((task_id / (total_wd * total_batch_size)))
remaining_task_id=$((task_id % (total_wd * total_batch_size)))
wd_index=$((remaining_task_id / total_batch_size))
remaining_task_id=$((remaining_task_id % total_batch_size))
batch_size_index=remaining_task_id
lr_value=${lr[$lr_index]}
wd_value=${wd[$wd_index]}
batch_size_value=${batch_size[$batch_size_index]}
srun python main.py --lr $lr_value --wd $wd_value --batch_size $batch_size_value --experiment_note $experiment_note --disease $disease
