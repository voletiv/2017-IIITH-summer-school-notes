# 20170705

## [Karteek Alahari](https://lear.inrialpes.fr/people/alahari/) (INRIA Grenoble) - Semantic Segmentation (13:30 - 14:30)

- Papazoglou et al., ICCV 2013

- EM-Adapt [Papandreou et al., 2015](https://arxiv.org/abs/1502.02734)

- M-CNN [Tokmakov et al., ECCV 2016](https://arxiv.org/abs/1603.07188)
    - Weakly-supervised semantic segmentation with motion cues
    - Video + Label -> FCNN -> Category appearance, Motion segmentation -> GMM -> Foreground appearance, (Category, foreground) -> Graph-based inference -> Inference labels
    - Better than Papazoglou et al.’s, better than EM-Adapt
    - Fine-tuning by re-training with intersection of outputs of EM-Adapt & our M-CNN
    - Pathak ICLR, Pathak ICCV and Papandreou use much more amount of data, but we achieve more accuracy in Weak Supervision (to be fair, they are pure weak supervision)

- Of course, now there are better standards to compare with

### LEARNING MOTION PATTERNS IN VIDEO [ArXiv Tech. rep. 2016]

- FlyingThings dataset, Mayer et al., CVPR 2016: synthetic videos of objects in motion

- Summary of motion estimation, video segmentation

### [MP-Net](https://arxiv.org/abs/1612.07217) (Encoder-Decoder Network)
    - Optical flow -> Encoder -> Decoder -> Objects in motion
    - Encoder: allows a large spatial receptive field
    - Decoder: output at full resolution
    - Image from FlyingThings, Ground Truth optical flow -> Motion segmentation

### [DAVIS Challenge [Perazzi et al., CVPR 2016]](http://davischallenge.org/) (Densely Annotated VIdeo Segmentation dataset)
    - Image -> Estimated Optical Flow (LDOF) -> Motion segmentation

- Optical flow can be computed using CNNs

- Try to capture what the “object” in the scene is

- Combine MP-Net prediction with “object-ness” to get better prediction (as a sort of post-processing)

- We can refine segmentation using a [Fully-connected CRF [Krahenbuhl and Koltun, 2011]](https://arxiv.org/abs/1210.5644)
    - Unary score + colour-based pairwise score

- Evaluation datasets:
    - FT3D (FlyingThings): 450 synthetic test videos, use ground truth flow
    - DAVIS: 50 videos
    - BMS (Berkeley Motion Segmentation): 16 real sequences corresponding to objects in motion

### [CRF as RNN [Zheng et al., 2015]](https://arxiv.org/abs/1502.03240)
    - Mean field inference iteration as a stack of CNN layers
