#!/bin/bash

#SBATCH --partition="kira-lab,overcap"
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="a40:8"
#SBATCH --qos="short"
#SBATCH --mem-per-gpu=45G
#SBATCH -x xaea-12,dave,randotron

export HOME="/coc/testnvme/chuang475"
export CONDA_BASE_PATH="${HOME}/miniconda3"
export CONDA_ENV_NAME="inspire-oft" # openvla rlds_env
export PYTHON_BIN="${CONDA_BASE_PATH}/envs/${CONDA_ENV_NAME}/bin/python"
export PRISMATIC_DATA_ROOT="/coc/testnvme/chuang475/datasets"
export HUGGINGFACE_HUB_CACHE="/coc/testnvme/chuang475/huggingface_cache"
export prismatic_data_root="data/prismatic"

cd /coc/testnvme/chuang475/projects/VLA-Adapter

num_gpus=8
num_processes=32
task_suite_names=(
    # "libero_90"
    # "libero_goal"
    # "libero_object"
    # "libero_spatial"
    "libero_10"
)

name="VLA-Adapter/LIBERO-Long-Pro"

# srun -u ${PYTHON_BIN} -m experiments.robot.libero.run_libero_eval \
#   --use_proprio True \
#   --num_images_in_input 2 \
#   --use_film False \
#   --pretrained_checkpoint runs/VLA-Adapter/LIBERO-Long-Pro \
#   --task_suite_name libero_10 \
#   --use_pro_version True \

for task_suite_name in "${task_suite_names[@]}"; do
    srun -u ${PYTHON_BIN} -m vla-scripts.parallel_libero_evaluator \
        --num-trials-per-task 50 \
        --num-gpus $num_gpus \
        --num-processes $num_processes \
        --task-suite-name $task_suite_name \
        --pretrained-checkpoint runs/$name \
        --save-root results/$name \
        --center-crop true \
        --use-l1-regression \
        --use-proprio \
        --num-images-in-input 2 \
        --use-pro-version \
        --use-minivlm \

done
