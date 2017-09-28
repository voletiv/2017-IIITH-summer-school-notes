
# 20170704

## [Chetan Arora](http://faculty.iiitd.ac.in/~chetan/) (IIIT Delhi) - Detection (09:30 to 12:20)

- Detection: Object Location, Object Attributes

- Applications: Instance Recognition, Assistive Vision, Security/Surveillance, Activity Recognition (in videos)

- Challenges: Illumination, occlusions, background clutter, intra-class variation

### Object Recognition Ideas: Historical
    - Geometrical era
        - Fit model to a transformation between pairs of features
        - Machine Perception of Three Dimensional Solids, a PhD Thesis in 1963
        - It’s invariant to camera position, illumination, internal parameters
        - Invariant to similarity Tx of four points
        - But, intra-class variation is a big problem
    - Appearance Models
        - Eigenfaces ([Turk and Pentland, 1991](http://www.face-rec.org/algorithms/PCA/jcn.pdf))
        - Other appearance manifolds
        - Requires global registration of patterns
        - Not even robust to translation, forget about occlusions, clutter, geometric Tx
    - Sliding Window
        - Like, Haar wavelet by Viola Jones on a sliding window ([Viola and Jones, 2001](http://www.vision.caltech.edu/html-files/EE148-2005-Spring/pprs/viola04ijcv.pdf))
        - THE method until deep learning came, because it’s pretty fast
        - But, we need non-maximal suppression
    - Local Features
        - SIFT (Lowe, 1999, 2004), SURF
        - Bag of Features: extract features -> bag them into words -> match stuff through words, not features
        - Spatial orientation is ignored
    - Parts-Based Models
        - Model: object as a set of parts, noting relative locations between parts, appearance of each part
        - [Fischer and Elschlager, 1973](http://dl.acm.org/citation.cfm?id=1309318)
        - Constellation model: recognizing larger parts in objects
        - Discriminatively-trained PBM: eg. HoG ([Ramanan, PAMI 2009](https://cs.brown.edu/~pff/papers/lsvm-pami.pdf)); can recognize object even if whole object can’t be seen

- Present ideas: global+local features, context-based, deep learning

### Object Detection with Deep Neural Networks

- ImageNet: excellent competition for image classification 

- [AlexNet (2012)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) won in 2012, changed the game

- CNN: Local spatial information is transmitted through the layers; fine-level information in channels, location-level in neurons themselves

- Convolutional layers are translation-invariant

- Responses from the layers are composed: one layer fires when it sees a head, a shoulder, the next layers fires when it sees both, etc.

- Convolutional layers can work with any image size, it’s just that the layers can be scaled accordingly

- HoG by Convolutional Layers:
    - HoG: Compute image gradients -> Bin gradients in 18 direction -> Compute cell histograms -> Normalize cell histograms
    - CNN: Learn edge filters -> Apply directional filters + gating (non-linearity) -> Sum/Average Pool -> LRN
    - So the same steps of HoG can be equivalently produced by a CNN, and even better because CNN can decide the bins, and other steps that were hitherto hand-engineered

- HoG, Dense SIFT, and many other “hand-engineered” features are convolutional feature maps

- Features in feature maps (one channel of any convolutional layer) can be back-tracked to a visual feature in the original image

- What if we apply CNNs to object classification + localization?

#### CLASSIFICATION + LOCALIZATION

- We could slide a window on the original image, input each image into, say, AlexNet, and find the window with max firing of object, say cat

- But Sliding Window approach in this case is computationally expensive, because too many windows (Viola-Jones, etc. are not computationally expensive)

- #### Localization as a regression problem:
    - Using AlexNet, in addition to a last layer to predict the object class, use another last layer to predict the 4 numbers of a bounding box
    - Total loss = Softmax loss of object class label + L2 loss of bounding box

- But this fails when there are multiple objects, or multiple instances of the same object!


- So let’s try to reduce the number of windows using Region Proposals

- #### [R-CNN](https://arxiv.org/abs/1311.2524):
    - Extract Region Proposals -> Resize each region to standard size -> Find Convolutional features -> Classify each region (maybe using SVM), and do linear regression for bounding box within proposed region
    - Training for R-CNN is not very accurate at first, and it is very slow
    - Ad-hoc training objectives (use an AlexNet pre-trained with ImageNet):
        - Fine-tune network with softmax classifier (log loss)
        - Train post-hoc linear SVMs (hinge loss)
        - Train post-hoc bounding box regressing (least squares)
    - But, this takes a looooong time to train, and a long time to infer (the class and bounding box)

- #### [SPP Net](https://arxiv.org/abs/1406.4729):
    - Instead of cropping the original image by the proposed regions and passing them through the network, pass the entire image, and then crop the extracted feature (say, at Conv5, the last convolutional layer in AlexNet) according to the region proposals!
    - But the extracted cropped-ish feature (feature cropping is not advisable, features are different from images) is of variable size depending on the size of the proposed region
    - So use Bag-of-Words and Spatial Pyramid Matching (SPM) to extract a uniform-sized vector at the end of the convolutional layers to input to the Dense layers
    - SPM: Make grids with pyramidally increasing size, pool the features into the grids, use these grids as the input to the Dense layers
    - This is what [SPP Net [He et al., ECCV14]](https://arxiv.org/abs/1406.4729) did
    - So SPP Net fixes the inference speed issue with R-CNN
    - But, we cannot pass gradients through the Spatial Pyramids! So link is broken, backprop can’t be used

- #### [Fast R-CNN](https://arxiv.org/abs/1504.08083):
    - Instead of spatial pyramids, use only 1 scale each, and do ROI Pooling
    - ROI Pooling layer is differentiable! So gradients can pass through it
    - Hierarchical Sampling: Choose shuffled ROIs from the SAME image in a minibatch while running SGD; this is because, while back-propagating, the weights corresponding to regions outside of the ROI also get updated, so it is better if we can update weights form the same image

- Also, in case of Fast R-CNN/SPP Net, the features at Conv5 have information from the surrounding area as well, while those in R-CNN don’t. This is a good thing, since the surrounding area provides context.

- So time is fine, all that’s left is Region Proposals

- #### [Faster R-CNN](https://arxiv.org/abs/1506.01497):
    - Anchors: Pre-defined reference boxes - different aspect ratios, and different scales
    - Propose regions from the feature map instead of the image! Using a separate convolutional network called Region Proposal Network for this
    - At each pixel, generate suggestions for box coordinates (x, y, w, h), based on the anchors, such that an object is present within that box; use IoU to select and reject boxes
    - Extract the most probable boxes containing objects using the Region Proposal Network
    - Use THESE proposed regions, do ROI Pooling and the rest to extract objects and their bounding boxes

- #### [YOLO](https://arxiv.org/abs/1506.02640):
   - Divide the image into 7x7 cells
    - At each cell, use a single network to generate 2 bounding boxes (based on anchors) and class probabilities (instead of an RPN like in Faster R-CNN)
    - Use the ground truth bounding box to increase the probability of the bounding box closer to it, and decrease that of the other one
    - Use Non-Maximal Suppression to eliminate boxes
    - But, multi-scale is not taken care of

- #### [Single Shot Multi-box Detection](https://arxiv.org/abs/1512.02325) (SSD) (ECCV16):
    - Multi-scale feature maps
    - Data augmentation
    - Also, while training, assign GT box to all unassigned generated boxes with IoU > 0.5
