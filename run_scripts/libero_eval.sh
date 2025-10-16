#!/bin/bash

#SBATCH --partition="kira-lab,overcap"
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="a40:8"
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

num_gpus=8
num_processes=32
task_suite_names=(
    "libero_90"
    "libero_goal"
    "libero_object"
    "libero_spatial"
    "libero_10"
)

# name="runs/VLA-Adapter/LIBERO-Long-Pro"
# name="runs/VLA-Adapter/LIBERO-Goal-Pro"
# name="runs/VLA-Adapter/LIBERO-Object-Pro"
# name="runs/VLA-Adapter/LIBERO-Spatial-Pro"
# name="outputs/configs+libero_object_no_noops+b8+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--object----5000_chkpt"
# name="outputs/configs+libero_90_no_noops+b8+lr-0.0002--image_aug--VLA-Adapter--90----50000_chkpt"
# name="outputs/configs+libero_90_no_noops+b8+lr-1e-05--image_aug--VLA-Adapter--90----50000_chkpt" # 1
# name="outputs/configs+libero_90_no_noops+b8+lr-1e-05--VLA-Adapter--90----50000_chkpt"
# name="outputs/configs+libero_90_no_noops+b8+lr-0.0001+AdamW+wd-0.0+x-action_queries--image_aug--VLA-Adapter--90----50000_chkpt" # 2
# name="outputs/configs+libero_90_no_noops+b8+lr-1e-05+SPD+wd-1.0--image_aug--VLA-Adapter--90----50000_chkpt"
# name="outputs/configs+libero_90_no_noops+b8+lr-1e-05+SPD+wd-2.0+x-action_queries--image_aug--VLA-Adapter--90----50000_chkpt"
# name="outputs/configs+libero_90_no_noops+b8+lr-1e-05+SPD+wd-0.7+x-action_queries--image_aug--VLA-Adapter--90----50000_chkpt" # 8
# name="outputs/configs+libero_90_no_noops+b8+lr-0.0001+AdamW+wd-0.0+x-action_queries+freeze_vlm--image_aug--VLA-Adapter--90----50000_chkpt" # 3
# name="outputs/configs+libero_90_no_noops+b8+lr-0.0001+AdamW+wd-0.0+x-action_queries+freeze_vision--image_aug--VLA-Adapter--90----50000_chkpt" # 4
# name="outputs/configs+libero_90_no_noops+b8+lr-0.0001+AdamW+wd-0.0+x-action_queries+freeze_language--image_aug--VLA-Adapter--90----50000_chkpt" # 5
#  name="outputs/configs+libero_90_no_noops+b8+lr-1e-05+AdamW+wd-0.0+x-action_queries+lpft--image_aug--VLA-Adapter--90----50000_chkpt" # 6
# name="outputs/configs+libero_90_no_noops+b8+lr-0.0001+AdamW+wd-0.0+x-action_queries+unfreeze_last_llm_layer+freeze_language--image_aug--VLA-Adapter--90----50000_chkpt" # 7

# name="outputs/configs+libero_90_no_noops+b8+lr-0.0001+AdamW+wd-0.0+x-action_queries+freeze_vlm+unfreeze_last_llm_layer--image_aug--VLA-Adapter--90----50000_chkpt"
# name="outputs/configs+libero_90_no_noops+b8+lr-0.0001+AdamW+wd-0.0+x-action_queries+unfreeze_last_llm_layer+freeze_language+freeze_vision--image_aug--VLA-Adapter--90----50000_chkpt"
# name="outputs/configs+libero_90_no_noops+b8+lr-0.0001+AdamW+wd-0.0+x-action_queries+unfreeze_last_llm_layer+freeze_language+freeze_dino--image_aug--VLA-Adapter--90----50000_chkpt"

name="output_flash6/configs+libero_90_no_noops+b8+lr-0.0001+SPD+wd-0.7+x-action_queries--image_aug--VLA-Adapter--90----45000_chkpt"
# name="output_flash6/configs+libero_90_no_noops+b8+lr-0.0001+SPD+wd-0.7+x-action_queries+freeze_vision--image_aug--VLA-Adapter--90----50000_chkpt"
# name="output_flash6/configs+libero_90_no_noops+b8+lr-0.0001+SPD+wd-0.7+x-action_queries+robust_ft_layers-language_model--image_aug--VLA-Adapter--90----50000_chkpt"
# name="output_flash6/configs+libero_90_no_noops+b8+lr-0.0001+SPD+wd-0.7+x-action_queries+robust_ft_layers-vision_backbone--image_aug--VLA-Adapter--90----45000_chkpt"
# name="output_flash6/configs+libero_90_no_noops+b8+lr-0.0001+SPD+wd-0.7+x-action_queries+robust_ft_layers-vision_backbone.featurizer_early_llm_layers--image_aug--VLA-Adapter--90----50000_chkpt"
# name="output_flash6/configs+libero_90_no_noops+b8+lr-0.0001+SPD+wd-0.7+x-action_queries+unfreeze_last_llm_layer+freeze_language+robust_ft_layers-vision_backbone--image_aug--VLA-Adapter--90----50000_chkpt"
# name="output_flash6/configs+libero_90_no_noops+b8+lr-0.0001+SPD+wd-0.7+x-action_queries+unfreeze_last_llm_layer+freeze_language+robust_ft_layers-vision_backbone.featurizer--image_aug--VLA-Adapter--90----50000_chkpt"


# srun -u ${PYTHON_BIN} -m experiments.robot.libero.run_libero_eval \
#   --use_proprio True \
#   --num_images_in_input 2 \
#   --use_film False \
#   --pretrained_checkpoint runs/VLA-Adapter/LIBERO-Long-Pro \
#   --task_suite_name libero_10 \
#   --use_pro_version True \

for task_suite_name in "${task_suite_names[@]}"; do
    srun -u ${PYTHON_BIN} -m vla-scripts.parallel_libero_evaluator \
        --num-trials-per-task 10 \
        --num-gpus $num_gpus \
        --num-processes $num_processes \
        --task-suite-name $task_suite_name \
        --pretrained-checkpoint $name \
        --save-root results/osmesa/$name \
        --center-crop true \
        --use-l1-regression \
        --use-proprio \
        --num-images-in-input 2 \
        --use-pro-version \
        --use-minivlm \

done
