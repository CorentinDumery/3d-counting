# [ICCV25 Oral] Counting Stacked Objects

From multi-view images of a stack, count the total number of objects! 

[Project page](https://corentindumery.github.io/projects/stacks.html) |
[Dataset page](https://zenodo.org/records/15609540) |
[stack-dataset-generation repository](https://github.com/Noaemien/stack-dataset-generation)

## Installation

This repository was tested with torch `2.1.2+cu118` and nerfstudio `1.1.5` on Ubuntu 24.04.

1) Since our volume reconstruction is based on nerfstudio's implementation of 3DGS, you will need a nerfstudio environment. You can find [instructions here](https://docs.nerf.studio/quickstart/installation.html). Make sure you can run splatfacto before proceeding to the next steps: `ns-train splatfacto --data data/pasta`

Some fixes for common issues with nerfstudio:
-   To install `tiny-cuda-nn` it may be useful to downgrade g++:
```
sudo apt install gcc-11 g++-11
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
``` 
  *  If you're running this on a machine with limited CPU and the command fails when running `splatfacto` for the first time, it may help to pre-emptively build CUDA code with `pip install git+https://github.com/nerfstudio-project/gsplat.git`.

2) Install the `counting-splatfacto` method, which is simply `splatfacto` with a couple utilities added like saving accumulation and depth maps.
```
pip install -e .
ns-install-cli
```

Check you can run the new method: `ns-train counting-splatfacto --data data/pasta` 

3) Download [the density net weights](https://drive.google.com/file/d/1yvOVQu2dGoxsJIyX4PN-0f_tCRhZhLL-/view?usp=sharing) and [the weights for `depth_anything_v2_vitl.pth`](https://github.com/DepthAnything/Depth-Anything-V2/) and put them both in a `weights/` at the root of this repository.

4) Download DinoV2 and DepthAnythingV2 in `ext`. For example:
```
mkdir ext
cd ext
git clone https://github.com/facebookresearch/dinov2
cd dinov2
git checkout b48308a394a04ccb9c4dd3a1f0a4daa1ce0579b8 
pip install fvcore omegaconf
cd ..
git clone https://github.com/DepthAnything/Depth-Anything-V2/
mv Depth-Anything-V2 DepthAnythingV2
```

You can make sure that this step worked by running 
```
python counting_3d/utils/compute_depth.py --image-input data/pasta/images/frame_00001.jpg --image-output test.png
```

## Inference

Run the script:
`. process_scene.sh data/pasta`

You can also download more scenes from [our dataset](https://zenodo.org/records/15609540).

This repository brought a few minor changes to the original method, so in some cases, you may obtain results slightly different from the original paper.

## Data preparation

Important requirements:
- Each scene must have multi-view images, a reference measurement in centimeters to adjust for scale, and known volume of a single object v. If v is not known, it can be estimated using a separate set of images containing a single object and the same volume estimation used in `process_scene.sh`.
- This step will require the nerfstudio environment described above, plus:
```
pip install trimesh hydra scikit-image

mkdir weights
mkdir ext
cd ext
git clone git@github.com:facebookresearch/sam2.git
cd ..
```
- Download `sam2.1_hiera_large.pt` and put it in the `weights` folder


Data preparation: 
- Organize your data as follows: `path/to/folder/`, where folder contains `path/to/folder/input1`, `path/to/folder/input2`, etc, and `input1` contains all images for this scene.
- Write your `path/to/folder/` in script `data_preparation/data_prep.sh` and run it from the root of this repository: `. data_preparation/data_prep.sh` 
 - Important: to scale the cameras, we use a simple heuristic to measure a distance in 3DGS space, and compare it with your reference measurement. When 3DGS is training, use the provided UI to measure the distance in 3DGS that matches your reference measurement:
   - Open the viewer at `http://localhost:7007/`.
   - Once reconstruction is satisfying, click "Pause Training".
   - Press the Click button, then click at the start of the measurement.
   - Rotate the view, then press Click and click and the end of the measurement. The distance in camera scale is printed in the terminal.
   - Save that number, then interrupt the reconstruction with CTRL+C.
   - Enter this number and the real-world measurement when prompted.
 - Select the OBJECTS only with left click. Use right clicks for negative prompts. Then close the window, and the segmentation will be propagated to other frames.
 - If the objects are in a container, repeat this step for the container+objects. Segment both of them together, and close the window. Please make sure that the segmentations are satisfying before proceeding.

## Training

To be added soon...

## Dataset generation

Have a look at our [stack-dataset-generation repository](https://github.com/Noaemien/stack-dataset-generation).


## Citation

If this repository is helpful to you, please consider citing the associated publication:

```
@inproceedings{dumery2025counting,
   title = {{Counting Stacked Objects}},
   author = {Dumery, Corentin and Ett{\'e}, Noa and Fan, Aoxiang and Li, Ren and Xu, Jingyi and Le, Hieu and Fua, Pascal},
   booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
   year = {2025}
}
```

The [released dataset](https://zenodo.org/records/15609540) can also directly be cited with:
```
@misc{dumery2025stackcounting,
   title = {StackCounting Dataset},
   author = {Dumery, Corentin and Ett{\'e}, Noa and D'Alessandro, Adriano},
   year = {2025}
   publisher = {Zenodo},
   doi = {10.5281/zenodo.15609540},
   url = {https://doi.org/10.5281/zenodo.15609540},
}
```
