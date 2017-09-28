# 20170708

## [Dhruv Mahajan](https://research.fb.com/people/mahajan-dhruv/) - New Architectures in Deep Learning (09:00 to 12:30)

### [Batch Norm [Szegedy, 2015]](https://arxiv.org/abs/1502.03167)
    - They showed that Dropout could be replaced with this
    - But they showed that deeper networks perform worse than shallower ones, so:

### [Residual networks [He et al., 2015]](https://arxiv.org/abs/1512.03385)
    - They said deeper perform worse because of vanishing gradients
    - They added skip connections to eliminate this

- [Huang et al., 2016], observed that after training, removing a block hardly affects performance - meaning there is a lot of redundancy in ResNets

### [DenseNets [Huang et al., 2016]](https://arxiv.org/abs/1608.06993)
    - Dense block: Concatenate a layer’s output with every subsequent layer
    - Like a ResNet, cascade Dense blocks together
    - No. of parameters, flops required for similar accuracy is lesser than those for ResNets

### [Multi-Scale DenseNet [Huang et al., 2017]](https://arxiv.org/abs/1703.09844)
    - What if we could spend more time on harder images and lesser time on easier images?
    - Better results than even DenseNets

### [Hard Mixture of Experts [Gross et al., 2017]](https://arxiv.org/abs/1704.06363)
    - What about Large Scale learning?
    - Do a clustering at a layer, then pass (forward or backward) only through that path that belongs to the right cluster
    - Hierarchical Softmax
    - Differentiated Softmax

### Open problems
    - Incremental addition of classes

### CLASSIFICATION, DETECTION, SEGMENTATION - UNIFIED VIEW

### +Detection:

- [R-CNN](https://arxiv.org/abs/1311.2524): apply CNNs to proposed regions

- [Fast R-CNN](https://arxiv.org/abs/1504.08083): apply CNN to full image, split the features according to regions
    - Isn’t end-to-end, cannot be back-propagated

- [Faster R-CNN](https://arxiv.org/abs/1506.01497): use a network to propose regions
    - Is end-to-end

- But what about scale?

- [Feature Pyramid Networks [Lin et al., 2016]](https://arxiv.org/abs/1612.03144)
    - Featurized Image Pyramid: Scale input image, pass different scale to CNN: computationally prohibitive
    - Single Feature Map: Pass input image to CNN, scale the CNN features, then predict: prediction accuracy does not increase much
    - Pyramid feature hierarchy: Predict at each scale
    - FEATURE PYRAMID: Scale down the features, concatenate with upsampling of downscaled features and then predict at each scale

- [Fully Convolutional Networks [Long et al., 2016]](https://arxiv.org/abs/1411.4038)
     - Replace fc layers with conv layers
     - Introduce deconv layers for per-pixel prediction

- [PixelNet [Bansal et al., 2017]](https://arxiv.org/abs/1609.06694)
    - Neighbouring pixels are highly correlated, which breaks the SGD IID assumption
    - SO pick sparse random pixels from different images in a batch, track its progrss through the layers, concatenate the same-pixel features across layers, use that for SGD
    - This does not break the IID assumption, and gets better results

### +Instance Segmentation

- [Instance-Sensitive FCN [Dai et al., 2016]](https://arxiv.org/abs/1603.08678)
    - Make pixels translation variant by introducing position sensitive maps
    - But, this is agnostic to object categories
    - Segmentation and Detection are separate

- [Fully Convolutional Instance Aware Segmentation [Li et al., 2017]](https://arxiv.org/abs/1611.07709)
    - Compute scores for detection and segmentation
    - Integrated Detection and Segmentation to train both better

- [Mask R-CNN for Instance Segmentation [He et al., 2017]](https://arxiv.org/abs/1703.06870)
    - Extension of Faster R-CNN
    - Simultaneous Classification, Detection, Segmentation
    - Current state-of-the-art, both in accuracy and time
    - Pose estimation results on COCO

- So we know that combining different tasks leads to better results

- [UberNet: Universal Convolutional Net [Kokkinos, 2016]](https://arxiv.org/abs/1609.02132)
    - Step towards a universal net
    - Trains simultaneously with diverse datasets
    - Multiple (low, mid, high level) tasks
    - Memory efficient

### MULTI-MODAL: Image + Text

- Leverage text to understand images

### Image Captioning
    - Use pre-trained CNNs to get image embeddings
    - Use a projection matrix to get word embeddings
    - Learn an RNN and projection matrix to predict sentences, given image and word embedding as input

- [Unifying Visual-Semantic Embeddings [Kiros et al., 2014]](https://arxiv.org/abs/1411.2539)
    - CNN + LSTM to embed image, text -> Project image and word features to multimodal space )train: and get them closer) -> Get a sentence structure from the multi-modal space -> SL-NLM Decoder to get captions

- [Deep Visual-Semantic Alignments for Generating Image Descriptions [Karpathy, 2016]](http://cs.stanford.edu/people/karpathy/deepimagesent/)
    - Use RNN on image, get regions, associate with LSTM embedding for words

- [Show and Tell [Vinyals et al., 2016]](https://arxiv.org/abs/1411.4555)

- Other problems:
    - Visual QA
    - Social signals
    - Missing modalities (image without a caption)

### DEEP LEARNING FOR VIDEOS

- Facebook is a video-first company: video is the first-class media

- What is in the video? Recommendations, Objectionable content filtering

- Where is it in the video? Interesting portions, summarization, thumbnail

- Data evolution, classification, multi-modal

### Dataset evolution

#### Aspects:
    - Motion importance: long-term vs short-term motion modelling
    - Supervised vs Semi-supervised: annotations at frame-level or video-level
    - Video length: clips vs long videos
    - Source: wild or controlled

- Earlier datasets (2005): controlled environments

- [UCF-101 (2012)](http://crcv.ucf.edu/data/UCF101.php)
    - Action recognition dataset: 101 actions, unconstrained environment
    - Another example: HMDB

- [THUMOS (2014)](http://crcv.ucf.edu/THUMOS14/)
    - Untrimmed videos: we don’t know where the action was

- [Activity Net (2015)](http://activity-net.org/)
    - 200 action classes
    - Untrimmed video classification, temporal action proposal, temporal action localization
    - Activity Net Challenge

- [Kinetics (2017)](https://deepmind.com/research/open-source/open-source-datasets/kinetics/)
    - Large-scale dataset of YouTube videos for action recognition

- [Sports 1M (2014)](http://cs.stanford.edu/people/karpathy/deepvideo/)
    - Very large scale data set
    - Sports recognition, >1M videos, >5M clips, 438 classes of sports
    - Very challenging - because of such huge data

- YouTube-8M (2016)
    - 8M videos, >500,000 hours
    - 4800 diverse set of entities (not necessarily actions or sports)

### VIDEO FEATURE EXTRACTION

- Good aspects: scalable, …

- Community was split: hand-crafted vs deep (now everyone is going deep)

- Current best hand-crafted features:
    - iDT (Improved Dense Trajectories)
        - Dense sampling in spatial scales -> Motion modelling using Optical Flow -> Trajectory description at Vixel-level -> Feature extraction using HOG, HOF, MBH
        - Pros: No learning, don’t need large-scale training data
        - Cons: Highly hand-crafted, computationally intensive

- Two-Stream Convolutional Network [Simonyan and Zisserman, 2014]
    - Compute per-frame spatial features using a Spatial ConvNet, compute motion features using a Temporal ConvNet, fuse them together and predict
    - Comparable results to iDT

- But how to do the fusion of spatial and temporal features?

- When to do temporal fusion? [Karpathy et al., 2014](http://cs.stanford.edu/people/karpathy/deepvideo/)
    - Single Frame vs Late Fusion vs Early Fusion vs Slow Fusion
    - They found Slow fusion performs better, so there’s a sweet spot where we need to go deep in a temporal sense as well
    - I [Dhruv Mahajan] would say there exist datasets where Slow Fusion seems to work best, not that Slow Fusion works best in every case

- [Video LSTM [Srivastava et al., 2014]](https://arxiv.org/abs/1502.04681)
    - How about replacing late fusion with LSTMs?
    - Unsupervised
    - Video through LSTM -> 1) Image Reconstruction LSTM, 2) Future Prediction LSTM

- 2D ConvNet vs 3D ConvNet
    - Most prior work uses 2D ConvNets
    - But they don’t model temporal information well

- [C3D: Generic Deep Features for Videos [Tran et al., 2014]](https://arxiv.org/abs/1412.0767)
    - Train 3D ConvNets on large-scale supervised video datasets for feature learning
    - Use the output of the C3D network as features, for classification, detection, etc.
    - Good architectures for 3D ConvNet: on UCF-101, depth=3 performed best, while comparing depths of 1, 5, 7, and comparing increase or decrease of kernel width with depth
    - 8 conv, 5 pool, 2 fc, 3x3x3 conv kernels, 2x2x2 pooling kernels
    - Visualizing low-level filters
    - Visualizing computed features

- [Res3D: Better Architecture for Better Spatiotemporal Features [Tran et al.]](https://arxiv.org/abs/1708.05038)
    - Conduct careful architecture search along several dimensions: depth 18 is good enough
    - Frame sampling rate: sampling 1 frame every 2-4 frames is fine => 8 fps is more than enough
    - Good input resolution: 128x128 is enough
    - Types of convolutions: 2D vs 3D vs 2.5D (???) vs Mixed (3D early on, then 2D)

### VIDEO VOXEL PREDICTION

- Semantic segmentation of videos, optical flow prediction, b&w->colour

- [Deep End2End Voxel2Voxel Prediction [Tran et al., 2015]](https://arxiv.org/abs/1511.06681)
    - Architecture for semantic segmentation in videos
    - Results on GATECH data
    - Optical Flow: compared with Brox’s method

### MULTI-MODALITY: Video + Text

- Real world is multi-modal, take advantage of non-text information

- [Video to Natural Language via RNN [Venugopalan et al., 2015]](https://arxiv.org/abs/1412.4729)
    - CNNs for frame-level features -> RNNs to generate captions

- [Sequence To Sequence: Video to Text [Venugopalan, 2016]](https://arxiv.org/abs/1505.00487)
    - Feed two synchronous LSTMs, one fed with video while other with pads, then the other with text while the first with pads, and captions are predicted

### MULTI-MODALITY: Video + Audio

- [Greatest Hits Dataset [Owens et al., 2016]](http://vis.csail.mit.edu/)
    - 977 videos, people hit and scratch materials with a drumstick

- [Visually Indicated Sounds [Owens et al., 2016]](https://arxiv.org/abs/1512.08512)
    - Given video frames, use RNN to produce sound

- [Ambient Sound for Visual Learning [Owens et al., 2016]](https://arxiv.org/abs/1608.07017)
    - Use sound as a natural supervisory signal
    - Predict statistical summary of sound from videos using CNN
    - Observed: clustering on the basis of the sound gives good clustering
    - Observed: more detectors active (per class) for classes with characteristic sounds

- [Lip Reading Sentences in the Wild [Chung et al., 2017]](https://arxiv.org/abs/1611.05358)
    - Watch, Listen, Attend, and Spell architecture
    - Images->ConvNet->LSTM, Sound->MFCC->LSTM  -> LSTM->Attention->MFCC, back into LSTM
    - Predicts unseen sentence correctly
