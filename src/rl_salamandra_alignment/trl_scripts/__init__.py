"""TRL scripts for Reinforcement Learning Algorithms
    """
import importlib.resources as pkg_resources
import os
from typing import Literal, get_args

TRL_Algorithms = Literal[
    "ppo",
    "ppo_tldr",
    "rloo",
    "rloo_tldr",
    "alignprop",
    "chat",
    "cpo",
    "dpo_online",
    "dpo_vlm",
    "gkd",
    "kto",
    "orpo",
    "reward_modeling",
    "sft",
    "sft_vlm",
    "bco",
    "ddpo",
    "dpo",
    "sft_video_llm",
    "xpo",
    "grpo",
]


def get_script_path(
    algorithm_name: TRL_Algorithms
) -> str:
    """Get the absolute path to a python script file for a Reinforcement Learning algorithmn

    Args:
        algorithm_name (TRL_Algorithms): Name Reinforcement Learning algorithm

    Returns:
        str: Path to python script
    """
    rl_salamanda_alignment_package_path = str(
        pkg_resources.files('rl_salamandra_alignment'))
    if algorithm_name == "ppo":
        script_path = os.path.join("ppo", "ppo.py")
    elif algorithm_name == "ppo_tldr":
        script_path = os.path.join("ppo", "ppo_tldr.py")
    elif algorithm_name == "rloo":
        script_path = os.path.join("rloo", "rloo.py")
    elif algorithm_name == "rloo_tldr":
        script_path = os.path.join("rloo", "rloo_tldr.py")
    elif algorithm_name in get_args(TRL_Algorithms):
        script_path = f"{algorithm_name}.py"
    else:
        raise ValueError(f"Unvalid RL Algorithm: {algorithm_name}")

    return os.path.join(rl_salamanda_alignment_package_path, "trl_scripts", script_path)
