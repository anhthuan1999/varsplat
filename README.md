<!-- ## (CVPR'26) VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM -->

<p align="center">

  <h2 align="center">VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM</h2>
  <p align="center">
    <a href="https://anhthuan1999.github.io/"><strong>Anh Thuan Tran</strong></a>
    ·
    <a href="https://cs.gmu.edu/~kosecka/"><strong>Jana Košecká</strong></a>
</p>

<p align="center"><strong>CVPR 2026</strong></a>
<p align="center">
    Department of Computer Science, George Mason University
</p>
   <h3 align="center">

   [![arXiv](https://img.shields.io/badge/arXiv-2603.09673-b31b1b.svg)](https://arxiv.org/abs/2603.09673) [![ProjectPage](https://img.shields.io/badge/Project_Page-VarSplat-blue)](https://anhthuan1999.github.io/varsplat/) [![Checkpoints](https://img.shields.io/badge/Checkpoints-Download-orange)](https://drive.google.com/drive/folders/1CaPlNdTewEkZCk0nHnmV5LhBx_XzwebX?usp=sharing) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
  <div align="center"></div>
</p>

<!-- ## Description

**VarSplat** is an uncertainty-aware RGB-D SLAM system leveraging 3D Gaussian Splatting. VarSplat explicitly learns per-splat appearance variance σ² and, via the law of total variance, renders differentiable per-pixel uncertainty maps through efficient single-pass rasterization. This uncertainty is integrated across all three downstream stages:

- **Tracking**: Per-pixel uncertainty provides short-horizon reliability, improving frame-to-frame pose updates.
- **Registration**: Per-pixel reliability supports mid-range alignment between overlapping submaps.
- **Loop Detection**: Per-splat variances modulate submap similarity and correct long-range drift.

Experimental results on **Replica** (synthetic) and **TUM-RGBD**, **ScanNet**, **ScanNet++** (real-world) show that VarSplat improves robustness and achieves competitive or superior tracking, mapping, and novel view synthesis rendering compared to existing 3DGS-SLAM baselines. -->

## Setup

The code has been tested on Rocky Linux 8.10, Python 3.10.1, CUDA 12.6, A100 80GB

### Repository

Clone the repo with `--recursive` because we have submodules:

```bash
git clone --recursive git@github.com:anhthuan1999/varsplat.git
cd VarSplat
```

### Installation

Make sure that gcc and g++ paths on your system are exported:

```bash
export CC=<gcc path>
export CXX=<g++ path>
```

To find the gcc and g++ paths on your machine you can use `which gcc`.

Then setup environment from the provided conda environment file:

```bash
conda create -n varsplat python=3.10
conda activate varsplat
pip install -r requirements.txt
```

You will also need to install *hloc* for loop detection and 3DGS registration:
```bash
cd thirdparty/Hierarchical-Localization
python -m pip install -e .
cd ../..
```

## Usage

### Downloading the Datasets

We evaluate on Replica, TUM-RGBD, ScanNet, and ScanNet++ datasets. We also provide scripts for downloading Replica and TUM-RGBD in the `scripts` folder. Install git lfs before using the scripts by running `git lfs install`.

For reconstruction evaluation on **Replica**, we follow [Co-SLAM](https://github.com/JingwenWang95/neural_slam_eval?tab=readme-ov-file#datasets) mesh culling protocol. Please use their code to process the mesh first.

For downloading **ScanNet**, follow the procedure described [here](http://www.scan-net.org/).
**Note:** There are some frames in ScanNet with `inf` poses. We filter them out using the notebook `scripts/scannet_preprocess.ipynb`. Please change the path to your ScanNet data and run the cells.

For downloading **ScanNet++**, follow the procedure described [here](https://kaldir.vc.in.tum.de/scannetpp/).

The config files are named after the sequences used in our experiments.

### Checkpoints

Pre-trained checkpoints for all evaluated scenes are available for download:

| Dataset | Link |
|---------|------|
| Replica | [Download](https://drive.google.com/drive/folders/1I6OZGF9D0SflVfQPyExhNWCxuhB78xs8?usp=drive_link) |
| TUM-RGBD | [Download](https://drive.google.com/drive/folders/1Mqc1uDUGe4hzzkpbbiOzv3z0Psp_ow4C?usp=drive_link) |
| ScanNet | [Download](https://drive.google.com/drive/folders/1TJFbFuauQaHVdRKSgcFlPNW6Ahug_sca?usp=drive_link) |
| ScanNet++ | [Download](https://drive.google.com/drive/folders/1vUFlCtZqYSPQty3eis59gF5N5vMSfn_o?usp=drive_link) |

### Quick Start

Run VarSplat on a single scene:

```bash
# Replica
python run_slam.py configs/Replica/office0.yaml \
  --input_path <path_to_replica>/office0 \
  --output_path output/replica/office0

# TUM-RGBD
python run_slam.py configs/TUM_RGBD/rgbd_dataset_freiburg1_desk.yaml \
  --input_path <path_to_tum>/rgbd_dataset_freiburg1_desk \
  --output_path output/tum/freiburg1_desk

# ScanNet
python run_slam.py configs/ScanNet/scene0000_00.yaml \
  --input_path <path_to_scannet>/scene0000_00 \
  --output_path output/scannet/scene0000_00

# ScanNet++
python run_slam.py configs/scannetpp/8b5caf3398.yaml \
  --input_path <path_to_scannetpp>/8b5caf3398 \
  --output_path output/scannetpp/8b5caf3398
```

You can also configure input and output paths directly in the config YAML file.

### SLURM scripts

If you are running on SLURM cluster, you can run for all scenes in a dataset by running the corresponding script in the `scripts` folder.

Please note the evaluation of `depth_L1` metric requires reconstruction of the mesh, which in turn requires headless installation of open3d if you are running on a cluster.

<!-- ## Variance Configuration

VarSplat introduces a `var` section in the per-scene config files with the following parameters:

| Parameter | Description |
|-----------|-------------|
| `use_nlldepth` | Whether to include depth residuals in the NLL variance loss |
| `tau_track` | Temperature parameter τ for uncertainty weighting during tracking |
| `tau_registration` | Temperature parameter τ for uncertainty weighting during registration |
| `tau_lc` | Temperature parameter τ for uncertainty weighting during loop detection |
| `w_limit` | Clamping range for normalized variance weights |

Example configuration:
```yaml
var:
  use_nlldepth: True
  tau_track: 10.0
  tau_registration: 10.0
  tau_lc: 5.0
  w_limit: 3.0
``` -->

## Acknowledgement

Our implementation builds upon [LoopSplat](https://github.com/GradientSpaces/LoopSplat), [Gaussian-SLAM](https://vladimiryugay.github.io/gaussian_slam/index.html), and [MonoGS](https://github.com/muskie82/MonoGS). We thank the authors for their open-source contributions.

## Citation

If you find our paper and code useful, please cite us:

```bib
@inproceedings{tran2026varsplat,
  title   = {VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM},
  author  = {Tran, Anh Thuan and Kosecka, Jana},
  booktitle = {CVPR},
  year    = {2026}
}
```
