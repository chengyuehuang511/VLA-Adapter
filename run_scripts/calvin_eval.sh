#!/bin/bash

#SBATCH --partition="kira-lab,overcap"
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --mem-per-gpu=45G
#SBATCH -x xaea-12,dave,randotron,nestor,cheetah

export HOME="/coc/testnvme/chuang475"
export CONDA_BASE_PATH="${HOME}/miniconda3"
export CONDA_ENV_NAME="inspire-oft" # openvla rlds_env
export PYTHON_BIN="${CONDA_BASE_PATH}/envs/${CONDA_ENV_NAME}/bin/python"
export PRISMATIC_DATA_ROOT="/coc/testnvme/chuang475/datasets"
export HUGGINGFACE_HUB_CACHE="/coc/testnvme/chuang475/huggingface_cache"
export prismatic_data_root="data/prismatic"

cd /coc/testnvme/chuang475/projects/VLA-Adapter

# name="runs/VLA-Adapter/CALVIN-ABC-Pro"
name="outputs/configs+calvin_abc_rlds+b8+lr-0.0002+AdamW+wd-0+x-action_queries+lora-r64+dropout-0.0--image_aug--VLA-Adapter--CALVIN-ABC----40000_chkpt"

srun -u ${PYTHON_BIN} -m vla-scripts.evaluate_calvin \
  --pretrained_checkpoint ${name} \
#   --task_suite_name libero_10 \
#   --use_pro_version True \
#   --use_proprio True \
#   --num_images_in_input 2 \
#   --use_film False \
