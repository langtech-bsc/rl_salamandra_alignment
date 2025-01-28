import yaml
import os
import json
from typing import Union
from copy import deepcopy
from rl_salamandra_alignment.distributed_configs import get_distributed_config_path
from rl_salamandra_alignment.trl_scripts import get_script_path
from rl_salamandra_alignment import templates
from rl_salamandra_alignment import logger
from rl_salamandra_alignment.utils.general import unfold_dict


def try_load_config(config_file: str) -> dict:
    """
    Load a YAML configuration file.

    Parameters:
    config_file (str): Path to the configuration file.

    Returns:
    dict: Configuration dictionary.
    """
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

    except FileNotFoundError:
        logger.warning(f"Configuration file {config_file} not found.")
        config = {}
    except yaml.YAMLError as exc:
        logger.warning(f"Error in configuration file: {exc}")
        config = {}
    logger.info("Using the following configuration:")
    logger.info(
        json.dumps(config, indent=2)
    )
    return config


def generate_slurm_preamble(sbatch_args: dict) -> str:
    """Generate the preamble for a slurm job.

    Args:
        sbatch_args (dict): Arguments for Sbatch

    Returns:
        str: Preamble with all #SBATCHs filled
    """
    slurm_preamble = ""
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
    try:
        output_dir = config["execution"]["output_dir"]
        if not isinstance(output_dir, str):
            raise ValueError("'output_dir' must be a string")
        return output_dir
    except:
        logger.warning(
            f"'output_dir' must be specified in your config file under 'execution'")


def setup_macro_output_dir_tree(output_dir: str) -> None:
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


def _internal_dir_paths(output_dir: str, id: str):
    d = {
        k: os.path.join(output_dir, k, id)
        for k in ["wandb", "cache", "training"]
    }
    return d


def _internal_file_paths(output_dir: str, id: str):
    d = {
        "slrum_script_distrubuted_run": os.path.join(output_dir, "slurm_scripts", f"distributed_run_{id}.job"),
        "slurm_script_launch": os.path.join(output_dir, "slurm_scripts", f'launch_{id}.sh'),
        "slurm_output": os.path.join(output_dir, "slurm_logs", f"{id}_output.log"),
        "slurm_error": os.path.join(output_dir, "slurm_logs", f"{id}_error.log"),
        "config": os.path.join(output_dir, "configs", f"config_{id}.json")
    }
    return d


def setup_micro_output_dir_tree(
    output_dir: str,
    config: dict,
    id: str
) -> None:

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
    launch_script_path = internal_file_paths["slurm_script_launch"]
    filled_template = replace_in_template(
        filled_template,
        "LAUNCH_SCRIPT",
        launch_script_path
    )

    return filled_template


def get_script_args_string(script_args_dict: dict) -> str:
    script_args_string = [
        f"\t--{k} {v}"
        for k, v in script_args_dict.items()
    ]
    return "\n".join(script_args_string)


def generate_launch_script(
        output_dir: str,
        config: str,
        id: str
) -> str:

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
    print(">>>>", rl_dataset_path)
    # This line avoids
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


def generate_one_job(output_dir: str, config: dict, id: str) -> tuple:

    launch_script_string = generate_launch_script(output_dir, config, id)

    distributed_run_script_string = generate_distributed_run_script(
        output_dir,
        config,
        id,
    )

    internal_file_paths = _internal_file_paths(output_dir, id)

    with open(
        internal_file_paths["slrum_script_distrubuted_run"], "w"
    ) as f:
        f.write(distributed_run_script_string)

    with open(
        internal_file_paths["slurm_script_launch"], "w"
    ) as f:
        f.write(launch_script_string)

    return (
        internal_file_paths["slrum_script_distrubuted_run"],
        internal_file_paths["slurm_script_launch"]
    )


def generate_all_job_files(config: dict) -> list[tuple]:

    output_dir = get_output_dir(config)
    setup_macro_output_dir_tree(output_dir)

    unfolded_configs = unfold_dict(config)
    unfolded_configs_with_id = [
        (str(id).zfill(2), cfg)
        for id, cfg in enumerate(unfolded_configs)
    ]
    for id, cfg in unfolded_configs_with_id:
        setup_micro_output_dir_tree(output_dir, cfg, id)

    return [
        generate_one_job(output_dir, cfg, id)
        for id, cfg in unfolded_configs_with_id
    ]
