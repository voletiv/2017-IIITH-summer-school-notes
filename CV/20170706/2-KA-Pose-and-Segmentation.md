# 20170706

## [Karteek Alahari](https://lear.inrialpes.fr/people/alahari/) - Pose and Segmentation (11:40 to 12:30)

- Tokmakov et al., 2017

- Video -> Optical flow->Motion network, Appearance network -> Classifier

### STEREO

- Pre-CNN era overview:
    - Disparity between L and R views -> Depth cues
    - RGB -> Motion cues
    - Disparity + RGB -> Person detections
    - Person detection -> Pose estimation -> Pose masks
    - Depth cues + Motion cues + Pose masks -> Segmentation

- Problem: given the disparity between L and R:
    - estimate Pose (𝚯)
    - estimate pixel label (person)
    - estimate disparity parameters (𝛕): layers, the layered ordering of people

- Computation of this over all possible values is an NP-hard problem to solve

- So instead, Energy function = Unary term + Spatial Pair-wise energy term + Temporal Pair-wise energy term

- Spatial Pair-wise term = Disparity smoothness + Motion smoothness + Colour smoothness
