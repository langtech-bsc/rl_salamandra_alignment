

{
    "slurm" : "SBATCH_PARAMETERS",
    "wandb_project": "WANDB_PROJECT",
    "wandb_name": "WANDB_NAME",
    "launch_script" : None,
    "ds_config_path": None,
    "venv": "VENV_PATH",
    "rl_config_args": "RL_CONFIG_ARGS",
    "model_config_args": "MODEL_CONFIG_ARGS",
    "rl_script_args": "RL_SCRIPT_ARGS",
    "environment" : "ENVIRONMENT_VARIABLES"
}

EXAMPLE_CONFIG = {
    
    "execution" : {
        "algorithm": "dpo",
        "venv": "path_to_venv",
        "output_dir": "path_to_output_dir",
        "distributed_config": "DSZero3Offload"
    },
    "slurm": {
        "job-name": "rl_salamandra_alignment",
        "output": None,
        "error": None,
        "nodes" : 4,
        "cpus-per-task": 80,
        "gres": "gpu:4",
        "time": "2:00:00",
        "account": "bsc88",
        "qos": "acc_debug",
    },
    "rl_script_args" : {
        "dataset_name": "path_to_dataset"
    },
    "model_config_args": {
        "model_name_or_path": "path_to_model",
        "output_dir" : None,
        "attn_implementation": "flash_attention_2",
        "torch_dtype": "bfloat16"
    },
    "rl_config_args": {
        # RL configs are subclasses of transformers.TrainingArguments
        
        # Different RL algorithms have different uses of beta.
        # however, in most of them, it is the weight of the KL-divergence (Loss=reward+Beta*KL)
        "beta": 0.2,
        "max_length": 8192,
        "max_prompt_length" : 128, # Default. When especificed,you use the default data collator`
        "remove_unused_columns" : False,
        "dataset_num_proc": 1,
        # ====
        # From `TrainingArguments`:
        # ===
        "learning_rate":5.0e-6,
        "num_train_epochs": 2,
        "bf16": True,
        "eval_strategy" : "steps",
        "eval_steps": 0.05,
        
        "logging_dir": None,
        "local_rank": None,
        "report_to": "wandb",
        # These arguments help to manage GPU memory
        "per_device_train_batch_size": 2 ,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 8, 
        "gradient_checkpointing" :True,
        },
    "environment": {
        "WANDB_PROJECT" : "salamandra_alignment",
        "WANDB_NAME" : "test_alignment",
        "WANDB_DIR": None 
    }
}