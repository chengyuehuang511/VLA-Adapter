#!/bin/bash

#SBATCH --partition="kira-lab"
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

data_name=libero_90_no_noops

# lora
# srun -u ${PYTHON_BIN} -m torch.distributed.run \
#   --standalone \
#   --nnodes=1 \
#   --nproc_per_node=8 \
#   vla-scripts/finetune.py \
#   --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
#   --config_file_path pretrained_models/configs \
#   --data_root_dir data/modified_libero_rlds \
#   --dataset_name $data_name \
#   --run_root_dir outputs \
#   --use_film False \
#   --num_images_in_input 2 \
#   --use_proprio True \
#   --use_lora True \
#   --use_fz False \
#   --use_minivlm True \
#   --image_aug True \
#   --num_steps_before_decay 50000 \
#   --max_steps 50000 \
#   --save_freq 5000 \
#   --save_latest_checkpoint_only False \
#   --merge_lora_during_training True \
#   --batch_size 8 \
#   --grad_accumulation_steps 1 \
#   --learning_rate 2e-4 \
#   --lora_rank 64 \
#   --use_pro_version True \
#   --wandb_entity "chuang475-georgia-institute-of-technology" \
#   --wandb_project "$data_name" \
#   --run_id_note VLA-Adapter--90--$current_time \

# fft
# srun -u ${PYTHON_BIN} -m torch.distributed.run \
#   --standalone \
#   --nnodes=1 \
#   --nproc_per_node=8 \
#   vla-scripts/finetune.py \
#   --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
#   --config_file_path pretrained_models/configs \
#   --data_root_dir data/modified_libero_rlds \
#   --dataset_name $data_name \
#   --run_root_dir outputs \
#   --use_film False \
#   --num_images_in_input 2 \
#   --use_proprio True \
#   --use_lora False \
#   --use_fz False \
#   --use_minivlm True \
#   --image_aug True \
#   --num_steps_before_decay 50000 \
#   --max_steps 50000 \
#   --save_freq 5000 \
#   --save_latest_checkpoint_only False \
#   --merge_lora_during_training True \
#   --batch_size 8 \
#   --grad_accumulation_steps 1 \
#   --learning_rate 1e-5 \
#   --lora_rank 64 \
#   --use_pro_version True \
#   --wandb_entity "chuang475-georgia-institute-of-technology" \
#   --wandb_project "$data_name" \
#   --run_id_note VLA-Adapter--90--$current_time \
#   --optimizer SPD \
#   --weight_decay 7 \

# freeze_vlm
srun -u ${PYTHON_BIN} -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  vla-scripts/finetune.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir data/modified_libero_rlds \
  --dataset_name $data_name \
  --run_root_dir output_flash6 \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_lora False \
  --use_fz False \
  --use_minivlm True \
  --image_aug True \
  --num_steps_before_decay 50000 \
  --max_steps 50000 \
  --save_freq 5000 \
  --save_latest_checkpoint_only False \
  --merge_lora_during_training True \
  --batch_size 8 \
  --grad_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --lora_rank 64 \
  --use_pro_version True \
  --wandb_entity "chuang475-georgia-institute-of-technology" \
  --wandb_project "$data_name" \
  --run_id_note VLA-Adapter--90--$current_time \
  --optimizer FTP \
  --weight_decay 1e-2 \
  --freeze_vlm False \
  --freeze_language False \
  --unfreeze_last_llm_layer False \
  --freeze_vision False \
  --freeze_dino False \
  --ftp_k 0 \
  # --weight_decay_scheduler layerwise_decay \
  # --robust_ft_early_llm_layers True \
  # --robust_ft_layers "['language_model']" \
  # --robust_ft_layers "['vision_backbone']" \
  # --robust_ft_layers "['vision_backbone.featurizer']" \
  # --ftp_k 0 \
  # --lpft_path "outputs/configs+libero_90_no_noops+b8+lr-0.0001+AdamW+wd-0.0+x-action_queries+freeze_vlm--image_aug--VLA-Adapter--90----50000_chkpt" \

# resume
# srun -u ${PYTHON_BIN} -m torch.distributed.run \
#   --standalone \
#   --nnodes=1 \
#   --nproc_per_node=8 \
#   vla-scripts/finetune.py \
#   --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
#   --config_file_path outputs/configs+libero_90_no_noops+b8+lr-0.0001+SPD+wd-0.7+x-action_queries--image_aug--VLA-Adapter--90----15000_chkpt \
#   --data_root_dir data/modified_libero_rlds \
#   --dataset_name $data_name \
#   --run_root_dir outputs \
#   --use_film False \
#   --num_images_in_input 2 \
#   --use_proprio True \
#   --use_lora False \
#   --use_fz False \
#   --use_minivlm False \
#   --image_aug True \
#   --num_steps_before_decay 50000 \
#   --max_steps 50000 \
#   --save_freq 5000 \
#   --save_latest_checkpoint_only False \
#   --merge_lora_during_training True \
#   --batch_size 8 \
#   --grad_accumulation_steps 1 \
#   --learning_rate 1e-4 \
#   --lora_rank 64 \
#   --use_pro_version True \
#   --wandb_entity "chuang475-georgia-institute-of-technology" \
#   --wandb_project "$data_name" \
#   --run_id_note VLA-Adapter--90--$current_time \
#   --optimizer SPD \
#   --weight_decay 0.7 \
#   --freeze_vlm False \
#   --freeze_language False \
#   --unfreeze_last_llm_layer False \
#   --freeze_vision False \
#   --freeze_dino False \
#   --resume_vla_path outputs/configs+libero_90_no_noops+b8+lr-0.0001+SPD+wd-0.7+x-action_queries--image_aug--VLA-Adapter--90----15000_chkpt \
#   --resume True \
#   --resume_step 15000 \