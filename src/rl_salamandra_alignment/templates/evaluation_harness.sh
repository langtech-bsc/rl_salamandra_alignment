#!/bin/bash
{{SBATCH_PARAMETERS}}

# Print the SBATCH options in this file to have them logged
grep "#SBATCH" "$0" | head -n -1

# Go to Harness folder
cd /gpfs/projects/bsc88/mlops-lm-evaluation-harness/production/
git config --global --add safe.directory /gpfs/projects/bsc88/mlops-lm-evaluation-harness/production

# Save job ID to use in output filename
job_id=$SLURM_JOB_ID

model_path="{{MODEL_PATH}}"
model_name="{{MODEL_NAME}}"
tasks="{{TASK_LIST}}"
few_shot=5 # NOTE: For evaluation we always use 5-shot (Is it what we want for RL)
apply_chat_template=True # NOTES: Will have to be changed if we do RL on base models
fewshot_as_multiturn=True # NOTE: If `chat_template`, then we use multiturn

harness_eval_output_path={{HARNESS_EVAL_OUTPUT_PATH}}

echo "Model path: $model_path"
echo "Model name: $model_name"
echo "Tasks: $tasks"
echo "Few shot: $few_shot"
echo "Apply chat template: $apply_chat_template"
echo "Fewshot as multiturn: $fewshot_as_multiturn"
echo "Output path: $harness_eval_output_path"

# Singularity and HF configuration
echo "Loading singularity module..."
module load singularity
echo "Loaded singularity module."

export SINGULARITY_IMAGE_LLAMA_TAG="galtea-llmops-cuda_12.6.2_transformers_4.46.2.sif"
export SINGULARITY_IMAGE_NEMO_TAG="galtea-llmops-transformers_4.40.2.sif"
export SINGULARITY_IMAGES_DIR="/gpfs/projects/bsc88/singularity-images"

export PATH_CACHE="{{CACHE_DIR}}"
export HF_DATASETS_OFFLINE="1"
export HF_HOME="/gpfs/projects/bsc88/hf-home"
export LD_LIBRARY_PATH=/apps/ACC/CUDA/12.3/targets/x86_64-linux/lib/stubs/:$LD_LIBRARY_PATH
export SINGULARITY_CACHEDIR=$PATH_CACHE/cache_singularity
export SINGULARITY_TMPDIR=$PATH_CACHE/cache_singularity
export NUMBA_CACHE_DIR=$PATH_CACHE/cache_numba
export TORCHDYNAMO_SUPPRESS_ERRORS=True
export MLCONFIGDIR=/gpfs/scratch/bsc88/bsc088532/.cache/matplotlib
export NUMEXPR_MAX_THREADS=64
export VLLM_CONFIG_ROOT=$TMPDIR
export VLLM_CACHE_ROOT=$TMPDIR

# Prepare arguments for the lm_eval module
common_args="--tasks ${tasks} \
    --num_fewshot ${few_shot} \
    --batch_size 1 \
    --output_path ${harness_eval_output_path} \
    --log_samples \
    --seed 1234 \
    --model hf \
    --model_args pretrained=${model_path},trust_remote_code=True,parallelize=True \
    --trust_remote_code"

if [ "${apply_chat_template}" == "True" ]; then
    if [ "${few_shot}" == 0 ]; then
        common_args="$common_args --apply_chat_template"
    elif [ "${fewshot_as_multiturn}" == "True" ]; then
        common_args="$common_args --apply_chat_template --fewshot_as_multiturn"
    fi
fi

launch_command="python -m lm_eval ${common_args}"
echo "Evaluation launch command: ${launch_command}"

singularity run --nv --no-home -B $TMPDIR:/tmp "${SINGULARITY_IMAGES_DIR}"/"${SINGULARITY_IMAGE_LLAMA_TAG}" bash -c "${launch_command}"
