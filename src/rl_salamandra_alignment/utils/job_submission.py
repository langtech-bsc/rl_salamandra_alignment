"""Utilities for submitting jobs with SLURM
    """
import subprocess
    
def submit_job(job_file: str, dependency: str = None) -> str:
    """Function to submit a job and return its job ID
    Args:
        job_file (str): Path to Slurm job file
        dependency (str, optional): job id of dependency for submitting a new job. Defaults to None.

    Returns:
        str: job id of submitted job
    """

    command = ["sbatch", "--parsable"]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command.append(job_file)

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error submitting job: {result.stderr}")
        return None

    job_id = result.stdout.strip()
    return job_id


def submit_job_with_retries(
        job_file: str,
        n_retries: int
) -> str:
    """Submit a slurm job n_retries times with dependencies

    Args:
        job_file (str): Path to Slurm job file
        n_retries (int): number of retries

    Returns:
        str: job id of last submitted job
    """

    previous_job_id = None

    for i in range(1, n_retries):
        job_id = submit_job(job_file, dependency=previous_job_id)
        if job_id:
            print(f"Submitted job {i} with ID {job_id}")
            previous_job_id = job_id
        else:
            print(f"Failed to submit job {i}")
            break
    return job_id

