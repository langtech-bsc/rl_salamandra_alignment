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
# Convert dataset to expected format
# =======================================

# Note: Only one node converts the dataset

# Activate venv
{{LOAD_MODULES}}
{{VENV_DIR}}
{{SET_PYTHONPATH}}
source $VENV_DIR/bin/activate

export PATH_CACHE="{{CACHE_DIR}}"
rm -rf $PATH_CACHE
mkdir -p $PATH_CACHE

# Dataset for RL
export RL_DATASET_PATH="{{RL_DATASET_PATH}}"
echo "Using dataset:"
echo $RL_DATASET_PATH
OLD_RL_DATASET_PATH=$RL_DATASET_PATH
export RL_DATASET_PATH="$PATH_CACHE/rl_dataset"

rl_salamandra_convert_dataset \
    --input_path \
    $OLD_RL_DATASET_PATH \
    --output_path \
    $RL_DATASET_PATH


# =======================================
# Execute the launch script on every node
# =======================================

export LAUNCH_SCRIPT="{{LAUNCH_SCRIPT}}"

chmod +x $LAUNCH_SCRIPT
srun \
    --wait=500 \
    $LAUNCH_SCRIPT

# clean up
rm -rf $PATH_CACHE
