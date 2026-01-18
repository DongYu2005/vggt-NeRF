
# VGGT-NeRF: Visual Geometry Grounded Transformer for Generalizable NeRF

## [VGGT Project Page](https://vgg-t.github.io) | [MVSNeRF Original Project](https://apchenstu.github.io/mvsnerf/)

This repository contains the implementation of **VGGT-NeRF**, which improves upon the MVSNeRF framework by introducing Visual Geometry Grounded Transformers for better global consistency and detail recovery in generalizable radiance field reconstruction.

<img width="813" height="188" alt="image" src="https://github.com/user-attachments/assets/6a8ae3d6-ae19-462e-939a-9e859f57ad60" />

**Comparison between MVSNeRF and VGGT-NeRF. VGGT-NeRF demonstrates superior global consistency and depth estimation quality.**

## Installation

### Environment Setup
The code is tested on **Ubuntu 20.04 + Pytorch 1.10.1**.

```bash
conda create -n vggtnerf python=3.8
conda activate vggtnerf
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f [https://download.pytorch.org/whl/cu113/torch_stable.html](https://download.pytorch.org/whl/cu113/torch_stable.html)
pip install pytorch-lightning==1.3.5 imageio pillow scikit-image opencv-python configargparse lpips kornia warmup_scheduler matplotlib test-tube imageio-ffmpeg
````

## Dataset Preparation

### DTU Dataset

Please download the processed DTU dataset from the link below and unzip it into the `data/` folder.

- **Download Link:** [[dtu数据集_数据集-飞桨AI Studio星河社区](https://aistudio.baidu.com/datasetdetail/28716)]
    
- **Path Structure:**
    
    Plaintext
    
    ```
    My_VGGT_Project/
	├── configs/
	├── ...
	└── data/
	    └── dtu/
	        ├── Cameras/       # Camera parameters
	        ├── Depths/        # Depth maps
	        ├── Rectified/     # Input images
	        └── lists/         # Train/test split lists
    ```

## Model Checkpoints

We provide pre-trained models and fine-tuned checkpoints. Please download them and place them in the root directory (or unzip the provided package).

- **Download Link:** [权重链接](https://disk.pku.edu.cn/link/AAB4C933556FFB4C2EB7085E240FE6AACC)

After extraction, ensure you have the following folders:
**Directory Structure:**
Plaintext
```
My_VGGT_Project/
├── ckpts/
│   └── VGGT-1B                                   # Foundation Model
│
├── runs_new/
│   └── vggt_mvs_general_train_v1/
│       └── ckpts/
│           └── epoch=xx/                         # e.g., epoch=09
│               └── last-v3.ckpt                  # General Training Weight
│
└── runs_fine_tuning/
    └── scan1/
        └── vggt_finetune_scan1/
            └── ckpts/
                └── latest.tar                    # Fine-tuned Weight for Scan1
```
## Training

### 1. Generalization Training (Pre-training)

To train the generalized model across scenes.

**Option A: Train from scratch**

Bash

```
python train_vggt_nerf_pl.py --config configs/vggt_general.txt
```

**Option B: Resume training or Load checkpoint** If you want to resume from a specific checkpoint, add the `--ckpt` argument:

Bash

```
python train_vggt_nerf_pl.py \
    --config configs/vggt_general.txt \
    --ckpt './runs_new/vggt_mvs_general_train_v1/ckpts/{epoch:02d}/last-v3.ckpt'
```

_Note: Replace `{epoch:02d}` with the actual epoch folder name if needed._

### 2. Fine-tuning

To fine-tune the model on a specific scene (e.g., Scan 2) for higher quality reconstruction.

Bash

```
python train_vggt_nerf_finetuning_pl.py \
    --config configs/vggt_finetune.txt \
    --datadir ./data/dtu/scan2
```

- **--datadir**: Change this path to fine-tune on different scans (e.g., `./data/dtu/scan4`).
    
- **--config**: Configuration for fine-tuning is loaded from `configs/vggt_finetune.txt`.

## Rendering & Visualization

After training or fine-tuning, you can use the provided notebooks or scripts to render videos.

- Use `renderer_video.ipynb` to generate free-viewpoint videos.
-
| Rabbit Scene | Cup Scene |
| :---: | :---: |
| <img src="asset/rabbit.gif" width="100%"> | <img src="asset/cup.gif" width="100%"> |
| *VGGT-NeRF on Rabbit* | *VGGT-NeRF on Cup* |
## Citation

If you use this code, please consider citing the original MVSNeRF paper and the VGGT project:

Code snippet

```
@article{chen2021mvsnerf,
  title={MVSNeRF: Fast Generalizable Radiance Field Reconstruction from Multi-View Stereo},
  author={Chen, Anpei and Xu, Zexiang and Zhao, Fuqiang and Zhang, Xiaoshuai and Xiang, Fanbo and Yu, Jingyi and Su, Hao},
  journal={arXiv preprint arXiv:2103.15595},
  year={2021}
}
```

## Acknowledgements

This code is built upon [MVSNeRF](https://github.com/apchenstu/mvsnerf). We thank the authors for their excellent work.

## Development Log & Technical Reflections (项目开发日志与思考)

This project is not just a reproduction of MVSNeRF, but an exploration of introducing Transformer-based features (VGGT) into the NeRF pipeline. Below is a summary of our development process, architectural decisions, and the lessons learned from failed experiments.

### 1. The Core Pipeline & Innovation
Our initial goal was to enhance the generalizability of NeRF by replacing standard CNN features with **VGGT (Visual Geometry Grounded Transformer)** features.
* **Pipeline:** `Images -> VGGT Feature Extraction -> 3D Projection -> NeRF MLP -> RGB & Density (σ)`.
* **Innovation:** unlike MVSNeRF which relies on local CNN features, VGGT introduces global attention mechanisms. This allows the model to "hallucinate" correct geometry in textureless regions (e.g., white walls or tables) by attending to global context, solving the "holes" and "noise" problems often seen in MVS methods.

### 2. The Struggle: Speed vs. Accuracy (Exploration of Loss Calculation)
During development, we faced a major bottleneck: **Ray-Casting is slow**. To solve this, we explored two different paradigms for loss calculation:

#### Phase 1: The "Shortcut" Attempt (3D Point Regression)
In an attempt to speed up training and bypass the heavy computation of volumetric rendering, we tried a radical **Direct 3D Point Regression** strategy:
* **Idea:** Directly predict the RGB and $\sigma$ of a 3D point without ray integration.
* **Result:** **Failed.** The model failed to converge on large scenes (BlendedMVS/DTU).
* **Failure Analysis:**
    1.  **Inefficient Sampling:** In large scenes, the valid surface occupies a tiny fraction of the 3D bounding box. Random sampling mostly hits "empty space," leading to severe class imbalance (too many negatives).
    2.  **Lack of Physical Constraints:** Ray-casting (Volumetric Rendering) implicitly models occlusion via transmittance ($T_i$). Without this integral, the network cannot distinguish between a point on the surface and a point hidden behind it.
    3.  **Projection Ambiguity:** A point in empty space (air) still projects to a valid pixel on the 2D feature map. Without the ray-marching constraint, the network confusingly maps "empty space coordinates" to "strong image features," leading to geometric collapse.

#### Phase 2: Return to Ray-Casting (The Correct Path)
We realized that the **Volumetric Rendering Equation** is not just a rendering technique, but a necessary physical constraint for learning 3D geometry from 2D images.
* We reverted to the standard **Ray-Casting** pipeline.
* **Optimization:** Although slower, this method ensures that the gradient back-propagates correctly through the accumulated transmittance, allowing the network to learn accurate depth and occlusion relationships.

### 3. Challenges & Solutions
* **Generalization vs. Speed:** We initially underestimated the difficulty of training a generalizable model on limited hardware. The "shortcut" (Point Regression) taught us that **geometry priors cannot be skipped**.
* **Feature Alignment:** Mapping 3D points to VGGT's Transformer features required careful handling of projection matrices to ensure the attention mechanism focused on the relevant geometric structures.

### 4. Lessons Learned
1.  **Respect the Physics:** In NeRF-like tasks, the integral process (ray-marching) is crucial. It provides the necessary "occlusion prior" that direct regression lacks.
2.  **Sampling Matters:** For 3D learning, *where* you sample is as important as *what* you learn. The sparsity of 3D space is a major challenge that requires strategies like hierarchical sampling (which MVSNeRF uses) rather than brute-force random sampling.
3.  **Transformers Need Grounding:** VGGT is powerful, but only when firmly grounded in the physical rendering process.
