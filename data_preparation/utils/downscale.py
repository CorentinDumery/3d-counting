import os
from PIL import Image
import sys

def downscale_images(input_folder, output_folder, downscale, force=False):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
   
    all_exist = all(os.path.exists(os.path.join(output_folder, filename)) for filename in image_files)
    if all_exist and not force:
        print("downscale.py: All downscaled images already exist. No processing needed.")
        return

    for filename in image_files:        
        img_path = os.path.join(input_folder, filename)
        with Image.open(img_path) as img:
            width, height = img.size
            img_resized = img.resize((width // downscale, height // downscale))

            output_path = os.path.join(output_folder, filename)
            img_resized.save(output_path)
            print(f"Image {filename} downscaled and saved to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()    
    parser.add_argument('path', type=str)
    parser.add_argument('--downscale', type=int, default=4, help="")
    parser.add_argument('--force', action='store_true', help="Force downscaling even if all images already exist")
    opt = parser.parse_args()

    #input_folder = sys.argv[1] # "data/kitchen_cvlab"
    input_folder = os.path.join(opt.path, "images")
    output_folder = os.path.join(opt.path, f"images_{opt.downscale}")

    downscale_images(input_folder, output_folder, opt.downscale, opt.force)
