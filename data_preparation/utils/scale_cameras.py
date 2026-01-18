
import argparse
import json

def scale_camera_transforms(camera_transforms, scaling_factor):
    """
    Scale the translation part of the camera transforms based on the computed scaling factor.
    
    Parameters:
    camera_transforms (list): List of camera transform matrices (c2w) 
    scaling_factor (float): The scaling factor to apply.
    
    Returns:
    list: Scaled camera transform matrices.
    """
    for camera in camera_transforms:
        #print(camera)
        # Apply the scaling factor to the translation part (fourth column) of the transform matrix
        camera[0][3] *= scaling_factor
        camera[1][3] *= scaling_factor
        camera[2][3] *= scaling_factor
        
    return camera_transforms

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Scale camera transforms")
    parser.add_argument('--data', type=str, required=True, help="Path to the input folder.")
    parser.add_argument('--ref_distance', type=float, required=True, help="Real-world distance between the selected points in meters.")
    parser.add_argument('--measured_distance', type=float, required=True, help="3D measured distance between the selected points in meters.")
    parser.add_argument('--make_path_png', action='store_true',help='Turn image path from jpg to png')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    folder_path = args.data
    scaling_factor = args.ref_distance / args.measured_distance 

    camera_data_json = folder_path + "/transforms.json"  

    # Load the camera data from JSON
    with open(camera_data_json, 'r') as f:
        camera_data = json.load(f)

    for i in range(len(camera_data['frames'])):
        # Apply the scaling factor to the translation part (fourth column) of the transform matrix
        camera_data['frames'][i]['transform_matrix'][0][3] *= scaling_factor
        camera_data['frames'][i]['transform_matrix'][1][3] *= scaling_factor
        camera_data['frames'][i]['transform_matrix'][2][3] *= scaling_factor

        if args.make_path_png:
            camera_data['frames'][i]['file_path'] = camera_data['frames'][i]['file_path'][:-4] + ".png" 

    with open(folder_path + '/scaled_camera_data.json', 'w') as f:
        json.dump(camera_data, f, indent=4)

    print(f"Camera data successfully scaled and saved to {folder_path + '/scaled_camera_data.json'}.")

    import trimesh

    # Read the point cloud
    pcd = trimesh.load(folder_path + "/sparse_pc.ply", process=False)

    # Scale the points
    pcd.vertices *= scaling_factor

    # Save the scaled point cloud
    save_path = folder_path + "/scaled_sparse_pc.ply"
    pcd.export(folder_path + "/scaled_sparse_pc.ply")
    print(f"Camera data successfully scaled and saved to {save_path}.")
