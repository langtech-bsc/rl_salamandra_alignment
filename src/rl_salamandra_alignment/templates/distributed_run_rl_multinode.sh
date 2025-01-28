#!/bin/bash
{{SBATCH_PARAMETERS}}

# For debugging:
#sbatch -q acc_debug launch.sh
#salloc -A bsc88 -q acc_debug -n 2 -c 80 --gres=gpu:4 -t 02:00:00

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
}

export GPUS_PER_NODE=4 # Architecture of MN5
export NNODES=$SLURM_NNODES
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export MASTER_PORT=$(find_unused_port)


# =======================================
# Execute the launch script on every node
# =======================================

export LAUNCH_SCRIPT={{LAUNCH_SCRIPT}}

chmod +x $LAUNCH_SCRIPT
srun \
    --wait=100 \
    $LAUNCH_SCRIPT
