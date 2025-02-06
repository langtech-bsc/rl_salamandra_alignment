"""Console script for rl_salamandra_alignment."""
import rl_salamandra_alignment
from argparse import ArgumentParser
from rl_salamandra_alignment.generate_scripts import generate_all_job_files
from rl_salamandra_alignment.utils.general import try_load_config
from rl_salamandra_alignment.utils.job_submission import submit_job, submit_job_with_retries

def main():
    """Reinforcement Learning for Salamandra on MN5"""
    parser = ArgumentParser()
    parser.add_argument(
        "config",
        help="YAML Config file for RL"
        )
    parser.add_argument(
        "no_slurm_submission",
        action="store_false",
        help="If you use this flag, jobs will NOT be submitted to slurm"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help= "activate debug mode"
    )
    args = parser.parse_args()

    config = try_load_config(args.config)
    all_job_files = generate_all_job_files(config)

    # Setup debug mode:
    if args.debug:
        args.no_slurm_submission = True

    if not args.no_slurm_submission:
        for distributed_run_script, launch_script in all_job_files:
            print("="*10)
            print(f"Distributed execution script:\n{distributed_run_script}")
            print(f"Launch script:\n{launch_script}")


if __name__ == "__main__":
    main()
