#!/bin/bash

# Enable/Disable different steps of the pipeline.
VOLUME_ESTIMATION=1
DENSITY_ESTIMATION=1
COMBINE_RESULTS=1

folder=$1
base_name=$(basename "$1")

# Initialize results folder
folder_masked=counting_results/${base_name}_masked
mkdir -p $folder_masked/images


if [ -d "${folder}/box_seg" ]; then
  cp "${folder}/box_seg/"* "$folder_masked/images"
else
  cp "${folder}/obj_seg/"* "$folder_masked/images"
fi

# Copy the sparse point cloud and transforms file
cp ${folder}/sparse_pc.ply $folder_masked/
cp ${folder}/transforms.json $folder_masked/

exp_name="masked"
MODEL_EXP_NAME=1_0

if [ $VOLUME_ESTIMATION -eq 1 ]; then
    path_3dgs="outputs/${base_name}/counting-splatfacto/${exp_name}/nerfstudio_models/step-000007999.ckpt"

    if [ -f "$path_3dgs" ]; then
        echo "3DGS already exists."
    else
        # Run splatfacto on masked images.
        # We disable all scaling / centering options.
        ns-train counting-splatfacto --data $folder_masked \
                                    --viewer.quit-on-train-completion True \
                                    --experiment-name $base_name \
                                    --timestamp ${exp_name} \
                                    --pipeline.model.background_color "random" \
                                    --max-num-iterations 8000 \
                                    nerfstudio-data \
                                    --auto-scale-poses False \
                                    --scale-factor 1 \
                                    --center_method none \
                                    --orientation_method none \
                                    --scene_scale 1.0 
    fi

    # Replace eval mode in config file so that ns-eval processes all images
    sed -i 's/^\(\s*eval_mode:\s*\)fraction/\1all/' outputs/${base_name}/counting-splatfacto/${exp_name}/config.yml

    # Produce accumulation and depth maps, necessary for voxel carving
    ns-eval --load-config outputs/${base_name}/counting-splatfacto/${exp_name}/config.yml --render-output-path $folder_masked

    # Voxel carving on 3DGS output to estimate volume
    thickness=$(jq -r '.avg_thickness // "0.01"' "$folder_colmap/heatmaps/scalars.json" 2>/dev/null || echo "0.01")
    folder_masked=counting_results/${base_name}_masked
    path_3dgs="outputs/${base_name}/counting-splatfacto/${exp_name}/nerfstudio_models/step-000007999.ckpt"
    python counting_3d/measure_volume.py --splatfacto_path $path_3dgs --save_path $folder_masked/ --box-thickness $thickness --cameras_path $folder_masked/transforms.json
fi

if [ $DENSITY_ESTIMATION -eq 1 ]; then
    # Identify best crop, compute depth map and apply volume occupancy model
    python counting_3d/inference_density.py --exp-name 1_0 --input-folder $folder --output-folder $folder_masked
fi


if [ $COMBINE_RESULTS -eq 1 ]; then
    basename_no_digit=$(echo "$base_name" | sed 's/[0-9]//g')
    unit=$(jq -r --arg key "$basename_no_digit" '.[$key] // empty' "data/unit_volumes.json")
    python counting_3d/combine_values.py --unit-volume $unit --path $folder_masked --exp-name $MODEL_EXP_NAME 
fi