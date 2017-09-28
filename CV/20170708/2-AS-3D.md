# 20170708

## [Avinash Sharma](https://sites.google.com/site/asharmaresearch/) - Recent Advances in 3D (13:30 to 15:30)

- Applications of depth, 3D: autonomous navigation, VR, scene understanding (3D mapping,...)

- Depth from Stereo (classical)

- Depth from Monocular Image [Liu et al., 2016](https://arxiv.org/abs/1502.07411)

- Depth from Multiple Images / Videos [Karsch et al;., 2012]

- Depth from X: Shading [Prados et al., 2006](https://hal.archives-ouvertes.fr/inria-00070704/document), Focus [Favaro et al., 2002](https://www.researchgate.net/profile/Paolo_Favaro/publication/225176433_Learning_Shape_from_Defocus/links/0912f50d024e92f08c000000.pdf)

- Depth from Active Sensing [Lanman et al., 2009](http://alumni.media.mit.edu/~dlanman/research/3DIM07/Lanman-SurroundLighting-CVIU.pdf): differences in structured lighting, IR lighting, laser scanning

### DEPTH FROM STEREO

- (Adopted from M. Pollefeys) Disparity between source images in the eyes causes 3D

- (S. Savarese) Stereo pair: Camera centres O1, O2; point in real world P; projections on camera of P p1, p2

- Steps: Camera calibration -> Rectify images -> Compute disparity and estimate depth

- (S. Fidler) Epipolar line; we see that epipolar lines are slanted

- Rectification: To straighten the epipolar lines; use Homography Tx to make image planes parallel to baseline

- (S. Lazebnik) So, post rectification, mapping of one point in L to only displacement in x in R.

- Disparity x - x’ per pixel x in L can be found by searching for x’ in R that minimizes the difference in intensities

- Also, Disparity x - x’ = B*f/z, where B is the baseline, f is the focal length, z is the distance from the camera centres to the point in real world

- Thus, depth z can be found from disparity

- Challenges: local intensity values differ in the two cameras, specularities, intensity values at multiple places, missing pixel values, repetitive patterns, occlusions, transparency, perspective distortion

- To overcome local challenges, go for Path-Level Cost Aggregation (Sum of Absolute Differences (SAD) in entire path, Normalized Cross Correlation (NCC), Mutual Information)

- Challenges: assumes constant depth within patch (invalid for depth discontinuities, slanted/non-planar surfaces), repetitive textures

- Dynamic Programming

- Improving correspondence using Global Optimization

- MRF
    - Energy minimization on top of a Markov Random Field
    - Through [Graph Cut [Boykov et al., PAMI 2001]](http://www.cs.cornell.edu/rdz/Papers/BVZ-pami01-final.pdf), or Belief Propagation

- Deep learning: use deep networks to estimate disparity

- [MC(Matching Cost)-CNN [Yann LeCun et al., 2015]](https://arxiv.org/abs/1409.4326)
    - Check if two patches correspond to each other using a CNN
    - Aggregate over adaptive set of patches

- [Efficient Deep Learning for Stereo Matching [Luo et al., 2016]](http://www.cs.toronto.edu/~aschwing/LuoEtAl_CVPR2016.pdf)
    - Replaced concat and fc layers with an inner product, used Right Image Patches instead of one patch
    - Directly compute disparity
    - Minimize cross entropy with GT, introduce smoothness
    - Very very less time and number of params compared to MC-CNN

### [DEPTH FROM MONOCULAR IMAGE [Fayao Liu et al., 2016]](https://arxiv.org/abs/1502.07411)

- A CRF energy function optimized via CNNs

### Recent advances in 3D SHAPE ACQUISITION and CLASSIFICATION

- Classical approach: Volumetric Stereo
    - Divide the real world into voxels, assume you have calibrated cameras

- [3D-R2N2 [Choy et al., ECCV 2016]](https://arxiv.org/abs/1604.00449)
    - Encoder -> 3-D LSTM -> Decoder to 3D image

- [3D ShapeNets [Wu et al., 2015]](http://3dshapenets.cs.princeton.edu/)
    - 3D Shape classification
    - High intra-class variation

- [FPNN: Field-Probing Neural Networks for 3D Data, Li et al., NIPS 2016](https://arxiv.org/abs/1605.06240)
    - Use Field-Probing layers instead of convolutional layers
    - Field-Probing layers perform much faster than CNNs

- Intrinsic Structure Learning: Laplacian Eigenvectors

- Intrinsic Learning: Sharma et al., 2011

- [Intrinsic Deep Learning: Geodesic CNN [Masci et al., 2015]](https://www.cv-foundation.org/openaccess/content_iccv_2015_workshops/w22/papers/Masci_Geodesic_Convolutional_Neural_ICCV_2015_paper.pdf)
