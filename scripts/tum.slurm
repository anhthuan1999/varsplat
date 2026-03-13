#!/bin/bash
#SBATCH --job-name=VarSplat-TUM
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

dataset="TUM_RGBD"
scenes=("rgbd_dataset_freiburg1_desk" "rgbd_dataset_freiburg1_desk2" "rgbd_dataset_freiburg1_room" "rgbd_dataset_freiburg2_xyz" "rgbd_dataset_freiburg3_long_office_household")
INPUT_PATH="<path_to_tum_data>"
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
