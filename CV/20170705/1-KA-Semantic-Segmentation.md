# 20170705

## [Karteek Alahari](https://lear.inrialpes.fr/people/alahari/) (INRIA Grenoble) - Semantic Segmentation (09:00 - 10:30)

- Working in INRIA Grenoble

- Camvid (Brostow et al., PRL’08): possibly the first autonomous driving effort

- Better is to do object classification (which object is present), and object detection (localizes objects to bounding boxes) (Some papers...)

- Even better is Semantic Segmentation, to achieve pixel-level labelling (Some papers...)

- Goal: pixels in, pixels out

- Eg. Monocular depth estimation (Liu et al., 2015), boundary detection (Xie & Tu, 2015)

- Long history: Liebe ^ Schiele, 2003; He et al., 2004, 

- More recently: Deeplab [Chen et al., 2015], Long et al, [Pathak et al],  CRF as RNN

### Higher order CRF

- Pose the problem of Semantic Segmentation onto a graph: let each pixel be represented as a node on a graph (4 or 8 neighbourhood), each pixel can carry a semantic label (road, or car, ...)

- For each assignment, there is a cost called Unary Potential
    - ### Unary potential
    - 𝜓_i(x_i), cost of the class label
    - ### TextonBoost [Shotton et al., ECCV 2006](http://jamie.shotton.org/work/publications/eccv06.pdf)
    - Each feature is denoted by a pair: (rectangle r, texton t) == whether a region r contains the texton t
    - Texton: convolve with a set of filters (Paper used Gabor filters), cluster them (pixel-wise?) to make a Texton Map
    - Feature response: count of texton instances

- Better: Use feature type as another element - (rectangle r, feature f, texton t), where f belongs to {SIFT, HOG, etc.}

- Comparison of Brostrow and Unary Potential: we see that using Unary Potential performs better
    - ### Pair-Wise Potential
    - 𝜓_{i,j}(x_i, x_j), cost of adjacent pixels not having the same label
    - Contrast-Sensitive Potts model: taking adjacency into account

- Texton Boost: Unary + Pairwise!

- Better results with +Pairwise than just unary
    - ### Segment-based Potentials

- Single segmentation: meanshift?
    - Very good at making superpixels
    - Not very good at fine-level segmentation

- Combine multiple segmentations: combine multiple levels of sensitivity in segmenting an image

- To do this, we introduce Clique Potential
    - ### Clique Potential
    - 𝜓_c(x_c), cost of a clique/superpixel
    - Cliques shall have higher cost if they have multiple labels in them
    - This encourages label consistency within a clique
    - One version: Robust $P^{N}$ model - 𝜓_c(x_c) = N_i(x_c)*(1/Q)*𝛾_max if N_i(x_c)<=Q, 𝛾_max otherwise
    - 𝛾_max is label inconsistency

- Even better results with +HO than just Unary + Pairwise

- Unary + Pairwise + HO: BMVC ‘09

- SO FAR: what objects are in the scene, where are the objects; what about, _how many objects_?

- Also, the HO result has missed thin objects

- Maybe one way is to do object detection and count the number of object instances

### Detector-driven Segmentation

- Imposes hard constraint, cannot recover from false detections
    - ### Detector Potential
    - Detector potential = min_{y_d}(Strength of detection hypothesis + inconsistency cost (e.g., for occluded objects)), where y_d is all the possible segmentations

- We need a smarter way of computing y_d

- So: Alpha expansion, Graph cuts make the cost function simpler

- Using Detector Potential is able to combine multiple sliding window detections to eliminate some boxes, and extract thin objects

- Comparison without and with combined sliding window detectors in PASCAL VOC 2009

- SO FAR: what objects, where are the objects, how many objects - through classical CRF-based methods

- New competition in CVPR 2017: Make PASCAL Great Again (PASCAL VOC ended in 2012)

- DPM, and improvements were used through 2007-2012 on PASCAL VOC, to get to 40$ mAP

- Post-competition, in 2013, Regionlets were used to jump up to 40% mAP directly

- Also, in the previous methods, Unary Potential was learnt through supervised learning, Clique Potential was unsupervised

- CNNs (obviously) changed the game

- CNN performs object classification, R-CNN does object detection; how to adapt for Semantic (pixel-level) Segmentation?

### Semantic Segmentation using Fully Convolutional Networks

- CONVOLUTIONALIZE: First up, convert CNNs + Fully connected -> Fully convolutional

- Use AlexNet, VGG, but replace the FC layers with CNNs

- In the second part, upscale the layers to get back a layer with image-size full resolution

- Append 1x1 convolutions with channel predictions

- Combining several scales:
    - combine _where_ (local, shallow) with _what_ (global, deep): fuse the features from different levels into a “deep jet”
    - use skip layers, skipping with stride (comparing 32, 16, 8, best is with 8 stride)

- Thus, pixel-level segmentation was achieved,

- But this required pixel-level ground truth for training

- Can we use weaker forms of supervision? Maybe bounding boxes, or just text tags (“cat”, dog”)

### Weakly-supervised methods

- MIL-based [Pathak et al., ICRWL’15]

- Image-level aggregation [Pinheiro & Collobert, CVPR’15]

- Constraint-based [Papandreou et al., ICCV’15; Pathak et al., ICCV’15] (e.g.: at least p% must have that label)

- Papandreou et al.: Not very good with weak supervision using p% constraint (GT bird -> predicted bird+plane example)
