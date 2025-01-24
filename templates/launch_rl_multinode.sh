#!/bin/bash
# =======================================
# Description
# =======================================
# 
# Trainer script for DPO on MN5  (using the TRL python package from HF). 
# Works on multinode and multigpu.
# Heavily adapted script for instruction tuning ()"run_train_salamandra40b.sh")
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

# Activate TRL environment
export ROOT_DIR="/gpfs/projects/bsc88/text/models/rlhf/alignment"
source $ROOT_DIR/use_venv.sh
export REPO_DIR="/gpfs/projects/bsc88/text/models/rlhf/trl/02_alignment_repo_luis"

# Main arguments for DPO: model, dpo dataset, output dir
export MODEL="/gpfs/projects/bsc88/text/models/instruction-tuning/models/checkpoints/salamandra40b_v0.1"

#export DPO_DATASET_PATH="/gpfs/projects/bsc88/data/01-alignment-raw-data/dpo_aya__sanity_check_dataset"
#export OUTPUT_DIRNAME="05_40b_quick_dpo" 
export DPO_DATASET_PATH="/gpfs/projects/bsc88/data/01-alignment-raw-data/hh_rlhf"
export OUTPUT_DIRNAME="05_40b_hh_rlhf_dpo"
export OUTPUT_DIR="$ROOT_DIR/outputs/$OUTPUT_DIRNAME" 
echo $OUTPUT_DIR

# =======================================
# 2. Setting up cache and wandb
# =======================================

# Manage Cache
export PATH_CACHE="$OUTPUT_DIR/cache"
mkdir -p $PATH_CACHE
export HF_HOME=$PATH_CACHE
export HUGGINGFACE_HOME=$PATH_CACHE
export HF_DATASETS_CACHE=$PATH_CACHE
export NUMBA_CACHE_DIR=$PATH_CACHE
export WANDB_CACHE_DIR=$PATH_CACHE
rm -rf $PATH_CACHE

# WANDB:
timestamp=$(date +"%Y%m%d-%H.%M.%S")
prefix="salamandra40_dpo_"$SLURM_JOB_ID"_"$timestamp
export WANDB_PROJECT="salamandra_alignment"
export WANDB_NAME=$prefix
export WANDB_MODE=offline
export WANDB_INIT_TIMEOUT=600
export WANDB_DIR="/gpfs/projects/bsc88/text/models/rlhf/trl/wandb_logs"
mkdir -p $WANDB_DIR
export WANDB_CONFIG_DIR=$WANDB_DIR/config

# =======================================
# 3. Environment variables
# =======================================


# Deepspeed config file
export deepspeed_path_to_config="$REPO_DIR/configs/ds_type3_config.json"

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
    --log-dir $OUTPUT_DIR/torch_run_logs
)

# =======================================
# 5. Arguments for Python script
# =======================================

# Convert dataset
OLD_DPO_DATASET_PATH=$DPO_DATASET_PATH
export DPO_DATASET_PATH="$PATH_CACHE/dpo_dataset"
python \
    /gpfs/projects/bsc88/text/models/rlhf/trl/05_40b_dpo/convert_dataset.py \
    --input_path \
    $OLD_DPO_DATASET_PATH \
    --output_path \
    $DPO_DATASET_PATH


# Arguments for python script
dpo_script_args=(
    # dataset for DPO
    --dataset_name $DPO_DATASET_PATH
)
dpo_config_args=(
    # DPO config is a subclass of transformers.TrainingArguments

    --beta 0.2 # DPO's beta hyperparameter for KL-divergence (Loss=reward+Beta*KL)
    --max_length 8192 #We need to get this to 8192
    --max_prompt_length 128 # Default. When especificed,you use the default data collator`
    --remove_unused_columns False 
    --dataset_num_proc 1 # Number of processes to use for processing the dataset.
    
    # from TrainingArguments 
    --learning_rate 5.0e-6 
    --num_train_epochs 2 
    --bf16 True
    --logging_dir $OUTPUT_DIR/logs
    --local_rank $SLURM_LOCALID
    --eval_strategy steps
    --eval_steps 0.05

    # These arguments help to manage GPU memory
    --per_device_train_batch_size 2 
    --per_device_eval_batch_size 2
    --gradient_accumulation_steps 8 
    --gradient_checkpointing True
    
    # wandb
    --report_to "wandb" 

)
model_config_args=(
    --model_name_or_path $MODEL
    --output_dir $OUTPUT_DIR 
    --attn_implementation flash_attention_2
    --torch_dtype bfloat16
)


# =======================================
# 6. Execution
# =======================================



# launch multinode multigpu execution
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun "${torchrun_distributed_args[@]}" \
    /gpfs/projects/bsc88/text/models/rlhf/trl/02_alignment_repo_luis/scripts/python/dpo.py \
    --deepspeed $deepspeed_path_to_config \
    "${dpo_script_args[@]}" \
    "${dpo_config_args[@]}" \
    "${model_config_args[@]}" 

# clean up
rm -rf $PATH_CACHE
printf "Done :)" 


########################
########################
########################
########################
########################

