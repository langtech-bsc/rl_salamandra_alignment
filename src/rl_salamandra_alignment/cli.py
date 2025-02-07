"""Console script for rl_salamandra_alignment."""
import rl_salamandra_alignment
import os
import logging
from argparse import ArgumentParser
import rl_salamandra_alignment
from rl_salamandra_alignment.generate_scripts import generate_all_job_files
from rl_salamandra_alignment.utils.general import try_load_config
from rl_salamandra_alignment.utils.job_submission import submit_job, submit_job_with_retries
from rl_salamandra_alignment import logger


def main():
    """Reinforcement Learning for Salamandra on MN5"""
    parser = ArgumentParser()
    parser.add_argument(
        "config",
        help="YAML Config file for RL"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="activate debug mode. Nothing will be submited to slurm"
    )
    parser.add_argument(
        "--no_evaluation",
        action="store_true",
        help="Skip evaluation, only train"
    )

    args = parser.parse_args()

    # Setup debug mode:
    if args.debug:
        args.no_evaluation = True
        rl_salamandra_alignment.setup_logging(level=logging.DEBUG)
        logger.debug("Debugging mode!")

    config = try_load_config(args.config)
    if not config.get("evaluation"):
        logger.warning(
            "No 'evaluation' found in config file. Evaluation scripts will NOT be generated"
        )
        args.no_evaluation = True

    # Generate job files

    all_job_files = generate_all_job_files(config)

    if args.debug:
        logger.debug("Scripts have been generated and can be found in:")
        logger.debug(
            os.path.dirname(
                all_job_files[0]["slrum_training_distributed_run"]
            )
        )
        return

    # Submit jobs:

    for script_dict in all_job_files:

        # submit training
        training_job_id = submit_job(
            script_dict["slrum_training_distributed_run"],
        )
        print(f"Submitted Training: {training_job_id}")

        # submit evaluation
        if args.no_evaluation:
            pass
        else:
            # harness
            harness_job_id = submit_job(
                script_dict["slurm_eval_harness_job"],
                dependency=training_job_id
            )
            print(
                f"Submitted Harness Eval: {harness_job_id} dependent on {training_job_id}")

            # local
            local_eval_job_id = submit_job(
                script_dict["slurm_eval_local_job"],
                dependency=training_job_id
            )
            print(
                f"Submitted Local Eval: {local_eval_job_id} dependent on {training_job_id}")


if __name__ == "__main__":
    main()
