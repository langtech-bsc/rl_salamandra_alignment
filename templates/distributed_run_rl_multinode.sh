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


# =======================================
# Distributed computing variables
# =======================================

# Function to find a random unused port
find_unused_port() {
    local port
    while true; do
        # Generate a random port number between 10000 and 65000
        port=$(($RANDOM % 55000 + 10000))
        
        # Check if the port is unused
        if ! lsof -iTCP:$port -sTCP:LISTEN &>/dev/null; then
            echo $port
            return 0
        fi
    done


export GPUS_PER_NODE=4 # Architecture of MN5
export NNODES=$SLURM_NNODES
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export MASTER_PORT=$(find_unused_port)
export MASTER_ADDR=$SLURM_LAUNCH_NODE_IPADDR

# Inspecting:
echo xxxxxxxxxxxxxxxxxxxxxxxxxxxx
echo master addr: ${MASTER_ADDR}
echo master port: ${MASTER_PORT}
echo num nodes: ${NNODES}
echo xxxxxxxxxxxxxxxxxxxxxxxxxxxx


# =======================================
# Execute the launch script on every node
# =======================================

export LAUNCH_SCRIPT={{LAUNCH_SCRIPT}}

chmod +x $LAUNCH_SCRIPT.sh
srun \
    --wait=100 \
    $LAUNCH_SCRIPT.sh
