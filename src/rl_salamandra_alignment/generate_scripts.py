import yaml
import os
import json
from typing import Union
from copy import deepcopy
from rl_salamandra_alignment.distributed_configs import get_distributed_config_path
from rl_salamandra_alignment.trl_scripts import get_script_path
from rl_salamandra_alignment import templates
from rl_salamandra_alignment import logger
from rl_salamandra_alignment.utils.general import (
    unfold_dict
)




def generate_slurm_preamble(sbatch_args: dict) -> str:
    """Generate the preamble for a slurm job.

    Args:
        sbatch_args (dict): Arguments for Sbatch

    Returns:
        str: Preamble with all #SBATCHs filled
    """
    slurm_preamble = ""
    
    # Always ask for 4 gpus
    sbatch_args["gres"] = "gpu:4"
    sbatch_args["cpus-per-task"] = 80
    
    # Fill "#SBATCH" arguments
    for arg_name, arg_value in sbatch_args.items():
        slurm_preamble += f"#SBATCH --{arg_name}={arg_value}\n"

    slurm_preamble += "\n"*2
    return slurm_preamble


def replace_in_template(
    text_template: str,
    variable_name: str,
    variable_value: Union[str, int, float],
) -> str:
    """Inside a template, replaces a variable name "{{MY_VAR}}" by a value

    Args:
        text_template (str): Template
        variable_name (str): Name of the variable in all caps.
        variable_value (Union[str, int, float]): Value of the variable

    Returns:
        str: template with the filled slot
    """

    return text_template.replace(
        r"{{"+variable_name+r"}}",
        variable_value
    )


def get_output_dir(config: dict) -> str:
    """Extract the field 'output directory' from a config dict, making sure it is a string

    Args:
        config (dict): execution config dict for experiment

    Raises:
        ValueError: Raised if the value is not a string (e.g. a list).

    Returns:
        str: Path to the output directory
    """

    try:
        output_dir = config["execution"]["output_dir"]
        if not isinstance(output_dir, str):
            logger.warning(f"Found 'output_dir of type {type(output_dir)}")
            raise ValueError("'output_dir' must be a string")
        return output_dir
    except:
        logger.warning(
            f"'output_dir' must be specified in your config file under 'execution'")


def setup_macro_output_dir_tree(output_dir: str) -> None:
    """Construct the directory tree for running an experiment.

    Args:
        output_dir (str): root directory for the outputs of the experiment.
    """

    os.makedirs(output_dir, exist_ok=True)

    subdirs = [
        "slurm_scripts",
        "slurm_logs",
        "wandb",
        "cache",
        "training",
        "configs"
    ]
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    return


def _internal_dir_paths(output_dir: str, id: str) -> dict:
    """Generate paths to directories used for the outputs of a subexperiment.

    Args:
        output_dir (str): root directory for the outputs of the experiment.
        id (str): Sub-experiment id

    Returns:
        dict: Dictionary with the paths to the directories for the subexperiment
    """
    d = {
        k: os.path.join(output_dir, k, id)
        for k in ["wandb", "cache", "training"]
    }
    return d


def _internal_file_paths(output_dir: str, id: str):
    """Generate paths to directories used for the outputs of a subexperiment.

    Args:
        output_dir (str): root directory for the outputs of the experiment.
        id (str): Sub-experiment id

    Returns:
        dict: Dictionary with the paths to the files for the subexperiment
    """
    d = {
        "slrum_training_distributed_run": os.path.join(output_dir, "slurm_scripts", f"distributed_run_{id}.job"),
        "slurm_script_launch": os.path.join(output_dir, "slurm_scripts", f'launch_{id}.sh'),
        "slurm_training_output": os.path.join(output_dir, "slurm_logs", id ,f"%j_%x_training.log"),
        "slurm_training_error": os.path.join(output_dir, "slurm_logs", id, f"%j_%x_training.err"),
        "slurm_eval_harness_output": os.path.join(output_dir, "slurm_logs", id ,f"%j_%x_eval_harness.log"),
        "slurm_eval_harness_error": os.path.join(output_dir, "slurm_logs", id, f"%j_%x_eval_harness.err"),
        "slurm_eval_local_output": os.path.join(output_dir, "slurm_logs", id ,f"%j_%x_eval_local.log"),
        "slurm_eval_local_error": os.path.join(output_dir, "slurm_logs", id, f"%j_%x_eval_local.err"),
        "config": os.path.join(output_dir, "configs", f"config_{id}.json")
    }
    return d


def setup_micro_output_dir_tree(
    output_dir: str,
    config: dict,
    id: str
) -> None:
    """Construct the directory tree for running a subexperiment.

    Args:
        output_dir (str): root directory for the outputs of the experiment.
        config (dict): execution config dict for subexperiment
        id (str): Sub-experiment id
    """

    # slurm scripts -> handled by 'generate_*_script' functions
    # slurm logs -> handled by 'generate_*_script' functions
    internal_dir_paths = _internal_dir_paths(output_dir, id)
    # wandb :
    os.makedirs(
        internal_dir_paths["wandb"], exist_ok=True
    )
    # cache
    os.makedirs(
        internal_dir_paths["cache"], exist_ok=True
    )
    # training
    os.makedirs(
        internal_dir_paths["training"], exist_ok=True
    )
    # configs
    internal_file_paths = _internal_file_paths(output_dir, id)
    with open(
        internal_file_paths["config"],
        "w"
    ) as f:
        json.dump(config, f, indent=2)


def generate_distributed_run_script(
        output_dir: str,
        config: dict,
        id: str,
) -> str:
    """Write the script for distributed execution of a subexperiment, by filling out the script template.

    Args:
        output_dir (str): root directory for the outputs of the experiment.
        config (dict): execution config dict for subexperiment
        id (str): Sub-experiment id

    Returns:
        str: Text of the script for distributed execution
    """

    filled_template = templates.get_distributed_run_template()
    internal_file_paths = _internal_file_paths(output_dir, id)

    # ============
    # Slurm preamble
    # ============

    # Automatically determine log file
    slurm_config = deepcopy(config["slurm"])
    output_dir = get_output_dir(config)
    slurm_config["job-name"] = id + "_" + slurm_config["job-name"]
    slurm_config["output"] = internal_file_paths["slurm_output"]
    slurm_config["error"] = internal_file_paths["slurm_error"]

    slurm_preamble = generate_slurm_preamble(slurm_config)
    filled_template = replace_in_template(
        filled_template,
        "SBATCH_PARAMETERS",
        slurm_preamble
    )

    # fill the Launch script
    launch_script_path = internal_file_paths["slurm_training_launch"]
    filled_template = replace_in_template(
        filled_template,
        "LAUNCH_SCRIPT",
        launch_script_path
    )

    return filled_template


def get_script_args_string(script_args_dict: dict) -> str:
    """Convert a python dictionary into a bash dictionary

    Args:
        script_args_dict (dict): python dictionary to convert

    Returns:
        str: bash dictionary
    """
    script_args_string = [
        f"\t--{k} {v}"
        for k, v in script_args_dict.items()
    ]
    return "\n".join(script_args_string)


def generate_launch_script(
        output_dir: str,
        config: dict,
        id: str
) -> str:
    """Write the script for launching a subexperiment, by filling out the script template.

    Args:
        output_dir (str): root directory for the outputs of the experiment.
        config (dict): execution config dict for subexperiment
        id (str): Sub-experiment id

    Returns:
        str: Text of the script for launching the subexperiment
    """

    filled_template = templates.get_launch_script_template()
    internal_file_paths = _internal_file_paths(output_dir, id)
    internal_dir_paths = _internal_dir_paths(output_dir, id)

    # ============
    # Environment variables
    # ============

    environment_dict = deepcopy(config["environment"])
    # Automatically determine WANDB_DIR
    environment_dict["WANDB_DIR"] = internal_dir_paths["wandb"]

    # Automatically determine dir for training (checkpoints)
    environment_dict["TRAINING_OUTPUT_DIR"] = internal_dir_paths["training"]

    # venv
    environment_dict["VENV_DIR"] = config["execution"]["venv"]

    # Generate export statements:
    environment_variables = [
        f'export {k}="{v}"'
        for k, v in environment_dict.items()
    ]
    environment_variables = "\n".join(environment_variables)

    filled_template = replace_in_template(
        filled_template,
        "ENVIRONMENT_VARIABLES",
        environment_variables
    )

    # ============
    # RL Dataset
    # ============
    rl_dataset_path = config["rl_script_args"]["dataset_name"]
    
    # The following line is to avoid using the original path to the RL dataset.
    # Remember that the RL dataset is converted to a format that can be locally run in MN5, 
    # and this conversion happens before launching the fine-tuning
    config["rl_script_args"]["dataset_name"] = "$RL_DATASET_PATH"

    filled_template = replace_in_template(
        filled_template,
        "RL_DATASET_PATH",
        rl_dataset_path
    )

    # ============
    # Cache
    # ============

    filled_template = replace_in_template(
        filled_template,
        "CACHE_DIR",
        internal_dir_paths["cache"]
    )

    # ============
    # Distributed config
    # ============
    ds_config_path = get_distributed_config_path(
        config["execution"]["distributed_config"]
    )
    filled_template = replace_in_template(
        filled_template,
        "DS_CONFIG_PATH",
        ds_config_path
    )

    # ============
    # RL script args
    # ============
    rl_script_args = deepcopy(config["rl_script_args"])
    filled_template = replace_in_template(
        filled_template,
        "RL_SCRIPT_ARGS",
        get_script_args_string(rl_script_args)
    )

    # ============
    # RL config args
    # ============
    rl_config_args = deepcopy(config["rl_config_args"])

    # Automatically determine logging dir for training
    rl_config_args["logging_dir"] = "$TRAINING_OUTPUT_DIR/logs"

    # This is very important for distributed distribution
    rl_config_args["local_rank"] = "$SLURM_LOCALID"

    # Name for WanDB syncing
    rl_config_args["run_name"] = "$WANDB_NAME"

    filled_template = replace_in_template(
        filled_template,
        "RL_CONFIG_ARGS",
        get_script_args_string(rl_config_args)
    )

    # ============
    # Model args
    # ============
    model_config_args = deepcopy(config["model_config_args"])

    # automatically determine training dir:
    model_config_args["output_dir"] = "$TRAINING_OUTPUT_DIR"

    filled_template = replace_in_template(
        filled_template,
        "MODEL_CONFIG_ARGS",
        get_script_args_string(model_config_args)
    )

    # ============
    # RL script path
    # ============

    rl_script_path = get_script_path(config["execution"]["algorithm"])

    filled_template = replace_in_template(
        filled_template,
        "RL_SCRIPT_PATH",
        rl_script_path
    )

    return filled_template


def generate_one_training_job(output_dir: str, config: dict, id: str) -> dict:
    """Generate the TRAINING slurm scripts for a subexperiment

    Args:
        output_dir (str): root directory for the outputs of the experiment.
        config (dict): execution config dict for subexperiment
        id (str): Sub-experiment id

    Returns:
        dict: paths to the distributed execution script and the launching script
    """
    
    # Generate the content of the scripts
    launch_script_string = generate_launch_script(output_dir, config, id)

    distributed_run_script_string = generate_distributed_run_script(
        output_dir,
        config,
        id,
    )

    # Save the content to file
    internal_file_paths = _internal_file_paths(output_dir, id)

    with open(
        internal_file_paths["slrum_training_distributed_run"], "w"
    ) as f:
        f.write(distributed_run_script_string)

    with open(
        internal_file_paths["slurm_script_launch"], "w"
    ) as f:
        f.write(launch_script_string)

    return {
        "slrum_training_distributed_run": internal_file_paths["slrum_training_distributed_run"],
        "slurm_script_launch" : internal_file_paths["slurm_script_launch"]
    }

def generate_eval_scripts_for_one_training(output_dir: str, config: dict, id: str) -> dict:
    """Generate the EVALUATION slurm scripts for a subexperiment

    Args:
        output_dir (str): root directory for the outputs of the experiment.
        config (dict): execution config dict for subexperiment
        id (str): Sub-experiment id

    Returns:
        dict: paths to the evaluation scripts
    """
    return {}

def generate_one_job_set(output_dir: str, config: dict, id: str) -> tuple:
    """Generate the slurm scripts for a subexperiment (both training and evaluation)

    Args:
        output_dir (str): root directory for the outputs of the experiment.
        config (dict): execution config dict for subexperiment
        id (str): Sub-experiment id

    Returns:
        dict: paths to the scripts for training and evaluation
    """
    
    job_set_paths = {}
    
    # First generate training job scripts
    training_scripts = generate_one_training_job(output_dir, config, id)
    job_set_paths.update(training_scripts)
    
    # Then generate evaluation job scripts
    evaluation_scripts = generate_eval_scripts_for_one_training(output_dir, config, id)
    job_set_paths.update(evaluation_scripts)
    
    return job_set_paths


def generate_all_job_files(config: dict) -> list[tuple]:
    """Generate the slurm scripts for an experiment

    Args:
        config (dict): execution config dict for experiment

    Returns:
        list[tuple]: for each subexperiment, paths to the distributed execution script and the launching script
    """

    output_dir = get_output_dir(config)
    
    # build the tree structure
    setup_macro_output_dir_tree(output_dir)

    # Generate the configs for all the subexperiments
    unfolded_configs = unfold_dict(config)
    
    # Give an ID to each subexperiment config
    unfolded_configs_with_id = [
        (str(id).zfill(2), cfg)
        for id, cfg in enumerate(unfolded_configs)
    ]
    # build the tree structure for each subexperiment
    for id, cfg in unfolded_configs_with_id:
        setup_micro_output_dir_tree(output_dir, cfg, id)

    # Generate all scripts
    return [
        generate_one_job_set(output_dir, cfg, id)
        for id, cfg in unfolded_configs_with_id
    ]
