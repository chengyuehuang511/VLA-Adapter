#!/bin/bash

#SBATCH --partition="kira-lab,overcap"
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="a40:1"
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

# name=minivla-libero-90
# name=minivla-libero90-prismatic
# name="baseline/prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90+vq+n1+b16+x7"
# name=minivla-inspire-libero-90
# name="inspire/prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90+vq+n1+b16+x7-vqa"
# name="inspire/prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90+vq+n1+b16+x7-vqa-vlm_ans"
# name="baseline/prism-qwen25-dinosiglip-224px-last-layer+0_5b+mx-libero-90+vq+n1+b16+x7"
# name="baseline/prism-qwen25-dinosiglip-224px-lm-head+0_5b+mx-libero-90+vq+n1+b16+x7"
# name="baseline/prism-qwen25-dinosiglip-224px-lm-head-new-tokens+0_5b+mx-libero-90+vq+n1+b16+x7"
# name="baseline/prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90+oft+n1+b16+x7"
## name="baseline/prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90+n1+b16+x7"
# name="baseline/prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90+vq+spd_10+n1+b16+x7"
# name="inspire/prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90+vq+spd_3+n1+b16+x7-vqa"
# name="VLA-Adapter/LIBERO-Long-Pro"
# steps="50000"
# steps="122500"

# for task_suite_name in "${task_suite_names[@]}"; do
#     srun -u ${PYTHON_BIN} -m vla_scripts.parallel_libero_evaluator \
#         --num-trials-per-task 10 \
#         --num-gpus $num_gpus \
#         --num-processes $num_processes \
#         --task-suite-name $task_suite_name \
#         --pretrained-checkpoint runs/$name \
#         --save-root results/$name \
#         --with-vqa false \
#         --steps $steps \
#         --center-crop true \
#         --use-l1-regression \
#         # --vlm-checkpoint pretrained/prism-qwen25-extra-dinosiglip-224px-0_5b
# done

srun -u ${PYTHON_BIN} -m experiments.robot.libero.run_libero_eval \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint runs/VLA-Adapter/LIBERO-Long-Pro \
  --task_suite_name libero_10 \
  --use_pro_version True \

# for task_suite_name in "${task_suite_names[@]}"; do
#     srun -u ${PYTHON_BIN} -m vla_scripts.parallel_libero_evaluator \
#         --num-trials-per-task 10 \
#         --num-gpus $num_gpus \
#         --num-processes $num_processes \
#         --task-suite-name $task_suite_name \
#         --pretrained-checkpoint runs/$name \
#         --save-root results/$name \
#         --with-vqa false \
#         --steps $steps \
#         --center-crop true \
#         --use-l1-regression \
#         --use-proprio \
#         --use-wrist-image \
#         --num-images-in-input 2 \
#         --use-pro-version \
#         # --vlm-checkpoint pretrained/prism-qwen25-extra-dinosiglip-224px-0_5b
# done
