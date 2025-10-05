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

data_name=libero_10_no_noops

srun -u ${PYTHON_BIN} -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  vla-scripts/finetune.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir data/modified_libero_rlds \
  --dataset_name $data_name \
  --run_root_dir outputs \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_lora True \
  --use_fz False \
  --use_minivlm True \
  --image_aug True \
  --num_steps_before_decay 100000 \
  --max_steps 100000 \
  --save_freq 5000 \
  --save_latest_checkpoint_only False \
  --merge_lora_during_training True \
  --batch_size 8 \
  --grad_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --lora_rank 64 \
  --use_pro_version True \
  --wandb_entity "chuang475-georgia-institute-of-technology" \
  --wandb_project "$data_name" \
  --run_id_note VLA-Adapter--long--$current_time \