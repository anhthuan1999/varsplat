#!/bin/bash
#SBATCH --job-name=VarSplat-ScanNetPP
#SBATCH --output=%x-%N-%j.out
#SBATCH --error=%x-%N-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-12:00:00
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --array=0-4

dataset="scannetpp"
scenes=("b20a261fdf" "8b5caf3398" "fb05e13ad1" "2e74812d00" "281bc17764")
INPUT_PATH="<path_to_scannetpp_data>"
OUTPUT_PATH="<output_path>"
CONFIG_PATH="configs/${dataset}"
EXPERIMENT_NAME="varsplat"
SCENE_NAME=${scenes[$SLURM_ARRAY_TASK_ID]}

echo "Job for dataset: $dataset, scene: $SCENE_NAME"
echo "Starting on: $(date)"
echo "Running on node: $(hostname)"

python run_slam.py "${CONFIG_PATH}/${SCENE_NAME}.yaml" \
                   --input_path "${INPUT_PATH}/${SCENE_NAME}" \
                   --output_path "${OUTPUT_PATH}/${dataset}/${EXPERIMENT_NAME}/${SCENE_NAME}" \
                   --group_name "${EXPERIMENT_NAME}"

echo "Job for scene $SCENE_NAME completed."
echo "Finished at: $(date)"
