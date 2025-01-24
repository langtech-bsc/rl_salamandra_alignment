#!/bin/bash
#SBATCH --job-name=40b_salamandra_dpo
#SBATCH --output=./logs/%x.log
#SBATCH --error=./logs/%x.err
#SBATCH --nodes 4
#SBATCH -c 80
#SBATCH --gres=gpu:4
#SBATCH --time=2:00:00
#SBATCH -A bsc88
#SBATCH --qos=acc_debug
#SBATCH --exclusive

# For debugging:
#sbatch -q acc_debug launch.sh
#salloc -A bsc88 -q acc_debug -n 2 -c 80 --gres=gpu:4 -t 02:00:00

# adapting script for instruction tuning
# "caller_train_salamandra40b.sh"
# /gpfs/projects/bsc88/text/models/instruction-tuning/it-chat-v1/caller_train_salamandra40b.sh

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export SLURM_CPU_BIND=none # This line accelerates training x4 in mn5

chmod +x launch_dpo_multinode.sh
srun --wait=100 launch_dpo_multinode.sh
