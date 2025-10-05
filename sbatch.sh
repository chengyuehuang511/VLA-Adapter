#!/bin/bash

name="train_libero_spatial"
# name="train_libero_object"
# name="train_libero_goal"
# name="train_libero_long"
# name="train_libero_90"

job_name="${name}_$(date +%Y%m%d_%H%M%S)"
output_dir="output/${job_name}"

mkdir -p "$output_dir"
sbatch --output="${output_dir}/slurm-%j.out" --error="${output_dir}/slurm-%j.err" run_scripts/${name}.sh