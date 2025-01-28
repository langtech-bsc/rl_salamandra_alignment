"""Script templates
    """
import importlib.resources as pkg_resources
import os

def get_distributed_run_template()-> str:
    """Get template for the script for distributed execution

    Returns:
        str: template for the script for distributed execution
    """
    rl_salamanda_alignment_package_path = str(pkg_resources.files('rl_salamandra_alignment'))

    script_path = os.path.join(
        rl_salamanda_alignment_package_path,
        "templates",
        "distributed_run_rl_multinode.sh"
    )

    with open(script_path, "r") as f:
        script_template = f.read()

    return script_template

def get_launch_script_template()-> str:
    """Get template for the script for executing the RL algorithm

    Returns:
        str: template for the script for executing the RL algorithm
    """
    rl_salamanda_alignment_package_path = str(pkg_resources.files('rl_salamandra_alignment'))
    
    script_path = os.path.join(
        rl_salamanda_alignment_package_path,
        "templates",
        "launch_rl_multinode.sh"
    )

    with open(script_path, "r") as f:
        script_template = f.read()

    return script_template

