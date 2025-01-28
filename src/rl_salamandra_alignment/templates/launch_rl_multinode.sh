#!/bin/bash
# =======================================
# Description
# =======================================
# 
# Trainer script for RL on MN5  (using the TRL python package from HF).
# Works on multinode and multigpu.
#
# =======================================
# Table of Contents
# =======================================
# 1. Initialization
# 2. Setting up cache and wandb
# 3. Environment variables
# 4. Arguments for multinode execution
# 5. Arguments for Python script
# 6. Execution
# =======================================

# =======================================
# 1. Initialization
# =======================================

{{ENVIRONMENT_VARIABLES}}

# Dataset for RL
export RL_DATASET_PATH={{RL_DATASET_PATH}}

# Activate TRL environment
source $VENV_DIR/bin/activate
echo "Output directory:"
echo $TRAINING_OUTPUT_DIR
echo "Dataset:"
echo $RL_DATASET_PATH

export MASTER_ADDR=$SLURM_LAUNCH_NODE_IPADDR

# Inspecting:
echo xxxxxxxxxxxxxxxxxxxxxxxxxxxx
echo "master addr: $MASTER_ADDR"
echo "master port: $MASTER_PORT"
echo "num nodes: $NNODES"
echo xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# =======================================
# 2. Setting up cache and wandb
# =======================================

# Manage Cache
export PATH_CACHE={{CACHE_DIR}}
mkdir -p $PATH_CACHE
export HF_HOME=$PATH_CACHE
export HUGGINGFACE_HOME=$PATH_CACHE
export HF_DATASETS_CACHE=$PATH_CACHE
export NUMBA_CACHE_DIR=$PATH_CACHE
export WANDB_CACHE_DIR=$PATH_CACHE
rm -rf $PATH_CACHE

# WANDB:
timestamp=$(date +"%Y%m%d-%H.%M.%S")

# WANDB_PROJECT should be defined in ENVIRONMENT_VARIABLES
# export WANDB_PROJECT="..."
# WANDB_NAME should be defined in ENVIRONMENT_VARIABLES
export WANDB_NAME="$WANDB_NAME"$SLURM_JOB_ID"_"$timestamp
export WANDB_MODE=offline
export WANDB_INIT_TIMEOUT=600
# WANDB_DIR should be defined in ENVIRONMENT_VARIABLES
# export WANDB_DIR="wandb"
mkdir -p $WANDB_DIR
export WANDB_CONFIG_DIR=$WANDB_DIR/config

# =======================================
# 3. Environment variables
# =======================================


# Deepspeed config file
export deepspeed_path_to_config="{{DS_CONFIG_PATH}}"

# avoid parallel tokenization: apparently is not compatible with multi-node python
# https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
export TOKENIZERS_PARALLELISM=true

# =======================================
# 4. Arguments for multinode execution
# =======================================

# Torchrun arguments for distributed execution
torchrun_distributed_args=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES 
    --rdzv_id $SLURM_JOB_ID 
    --rdzv_backend c10d 
    #--rdzv_backend static
    --rdzv-endpoint $MASTER_ADDR:$MASTER_PORT
    #--rdzv-endpoint localhost:$MASTER_PORT
    #--master-addr $PARENT
    #--master-port $MPORT
    #--node-rank $RANK
    #--rdzv-conf is_host=True
    --log-dir $TRAINING_OUTPUT_DIR/torch_run_logs
)

# =======================================
# 5. Arguments for Python script
# =======================================

# Convert dataset
OLD_RL_DATASET_PATH=$RL_DATASET_PATH
export RL_DATASET_PATH="$PATH_CACHE/rl_dataset"

rl_salamandra_convert_dataset \
    --input_path \
    $OLD_RL_DATASET_PATH \
    --output_path \
    $RL_DATASET_PATH


# Arguments for python script
rl_script_args=(

{{RL_SCRIPT_ARGS}}
)
rl_config_args=(
# RL configs are subclasses of transformers.TrainingArguments
{{RL_CONFIG_ARGS}}
)
model_config_args=(
{{MODEL_CONFIG_ARGS}}
)


# =======================================
# 6. Execution
# =======================================



# launch multinode multigpu execution
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun "${torchrun_distributed_args[@]}" \
    {{RL_SCRIPT_PATH}} \
    --deepspeed $deepspeed_path_to_config \
    "${rl_script_args[@]}" \
    "${rl_config_args[@]}" \
    "${model_config_args[@]}" 

# clean up
rm -rf $PATH_CACHE
printf "Done :)" 


########################
########################
########################
########################
########################

