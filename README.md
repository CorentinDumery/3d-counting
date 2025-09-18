# [ICCV25 Oral] Counting Stacked Objects

From multi-view images of a stack, count the total number of objects! 

[Project page](https://corentindumery.github.io/projects/stacks.html) |
[Dataset page](https://zenodo.org/records/15609540) |
[stack-dataset-generation repository](https://github.com/Noaemien/stack-dataset-generation)

**Please note**: _The current code is sufficient to run the entire pipeline on the data presented in the paper, but I plan to release additional code to help with data preprocessing and model training. However, I am currently on internship leave until December 2025. Feel free to contact me or open an issue in the meantime, and I will try my best upon my return._

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

To be added soon... 
We will provide detailed instructions and preprocessing scripts. Most importantly, the data structure needs to follow the structure of our released data and the cameras need to be at metric scale.

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
