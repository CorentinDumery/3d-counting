#!/bin/bash

# Define the base directory
BASE_DIR="data/mva"

# Prepare data
PREP_FOLDER=1
COMPUTE_CAMERAS=1
MATCH_IMAGES_TO_HIGHRES=1
SCALE_CAMERAS=1
COMPUTE_MASKS=1
REORG_FOLDERS=1

downscale_fact=8

# Loop over all directories in BASE_DIR
for folder in "$BASE_DIR"/*/; do
    base_name=$(basename "$folder")
    folder_colmap=$BASE_DIR/${base_name}_colmap
    folder_masked=$BASE_DIR/${base_name}_masked

    if [[ "$base_name" == *_colmap || "$base_name" == *_masked ]]; then
        echo "Skipping folder: $base_name"
        continue
    fi

    echo "Processing folder: $folder (Base name: $base_name)"

    if [ $PREP_FOLDER -eq 1 ]
    then
        # If it doesn't exist, move all images to $folder/images
        if [ ! -d "$folder/images" ]; then
            mkdir -p $folder/images 
            mv $folder/*.png $folder/*.jpg $folder/images/ 2>/dev/null
        fi
        python data_preparation/utils/downscale.py $folder --downscale $downscale_fact
    fi

    if [ $COMPUTE_CAMERAS -eq 1 ]
    then
        if [ -d "$folder_colmap" ]; then
            echo "Folder $folder_colmap already exists. Skipping processing."
        else

            mkdir $folder_colmap
            ns-process-data images --data $folder/images_$downscale_fact/ --output-dir $folder_colmap --no-gpu

            read -p "Are you satisfied with the results? (y/n): " user_input
            if [[ "$user_input" == "n" || "$user_input" == "N" ]]; then
                echo "Deleting $folder_colmap..."
                rm -rf "$folder_colmap"
            fi
        fi

    fi

    if [ $MATCH_IMAGES_TO_HIGHRES -eq 1 ]
    then
        # Unfortunately, ns-process-data renames images and changes them a bit, 
        # this uses a heuristic to map to original names since we need the full-res images too.
        if [ -f "$folder_colmap/mapping_to_fullres_names.json" ]; then
            echo "Mapping information already exists. Skipping."
        else
        
            python data_preparation/utils/find_identical_images.py $folder_colmap/images $folder/images_$downscale_fact --output $folder_colmap/mapping_to_fullres_names.json
        fi
    fi

    if [ $SCALE_CAMERAS -eq 1 ]
    then
        scale_json="$folder_colmap/scale_info.json"
        if [ -f "$scale_json" ]; then
            echo "Scale information already exists in $scale_json. Skipping."
        else
            # backup:
            cp $folder_colmap/transforms.json $folder_colmap/old_transforms.json
            cp $folder_colmap/sparse_pc.ply $folder_colmap/old_sparse_pc.ply
            # python unit_scale_cameras.py --data $folder_colmap/
            
            echo "IMPORTANT: SEE INSTRUCTIONS IN README.md. We will now scale the cameras to *metric* units. This is required for volume estimation in cubic meters. Please use the Click button to measure distances in the 3DGS viewer."

            ns-train counting-splatfacto --data $folder_colmap nerfstudio-data --auto-scale-poses False --scale-factor 1 --center_method none --orientation_method none --scene_scale 1.0

            echo "Please enter the distance measured in 3DGS:"
            read recon_distance 
            echo "(in meters) Please enter the reference distance measured IRL:"
            read ref_distance 
            
            python data_preparation/utils/scale_cameras.py  --data $folder_colmap/ --ref_distance $ref_distance --measured_distance $recon_distance --make_path_png 
                    
            echo "{\"recon_distance\": $recon_distance, \"ref_distance\": $ref_distance}" > "$scale_json"
            echo "Scale information saved to $scale_json."
        fi
    fi

    if [ $COMPUTE_MASKS -eq 1 ]
    then
        if [ -d "$folder_masked" ]; then
            echo "Folder $folder_masked already exists. Skipping processing."
        else
            mkdir $folder_masked
            mkdir $folder_masked/images
            mkdir $folder_masked/box_seg
            mkdir $folder_masked/obj_seg

            echo What segmentation shift do you want for $base_name?
            read shift

            python data_preparation/utils/compute_sam2_masks.py --data $folder_colmap --type objects --shift $shift
            
            read -p "Is there a box? (y/n): " user_input
            if [[ "$user_input" == "y" || "$user_input" == "Y" ]]; then
                python data_preparation/utils/compute_sam2_masks.py --data $folder_colmap --type box --shift $shift
                cp $folder_colmap/box_seg/* $folder_masked/images 
            else
                cp $folder_colmap/obj_seg/* $folder_masked/images 
            fi                
            
            cp $folder_colmap/scaled_camera_data.json $folder_masked/transforms.json
            cp $folder_colmap/scaled_sparse_pc.ply $folder_masked/sparse_pc.ply
        fi
    fi
    if [ $REORG_FOLDERS -eq 1 ]
    then
        echo "Reorganizing folders for $base_name"
        mv $folder/images $folder/images_full_resolution
        rm -rf $folder/images_$downscale_fact
        mv $folder_colmap/images $folder/images
        mv $folder_colmap/box_seg $folder/box_seg
        mv $folder_colmap/obj_seg $folder/obj_seg
        mv $folder_colmap/scaled_camera_data.json $folder/transforms.json
        mv $folder_colmap/scaled_sparse_pc.ply $folder/sparse_pc.ply
        mv $folder_colmap/mapping_to_fullres_names.json $folder/mapping_to_fullres_names.json

        # Replace all instances of .jpg to .png in transforms.json
        sed -i 's/\.jpg/\.png/g' $folder/transforms.json

        echo "Please enter the ground truth count for $base_name (or -1 if unknown):"
        read gt_count
        echo "Please enter the unit volume in cm3/mL for $base_name (or -1 if unknown):"
        read unit_volume

        echo "{\"gt_count\": $gt_count, \"unit_volume\": $unit_volume}" > "$folder/info.json"

        rm -rf $folder_colmap
        rm -rf $folder_masked
    fi
done