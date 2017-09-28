# 20170705

## [Gaurav Sharma](http://www.grvsharma.com/research.html) (IIT Kanpur) - Face and Action (11:00 - 12:30)

### FACES

- Motivation for studying faces, cameras, etc.

### Face Recognition

- Face Recognition evolution [Chellappa et al., Human and Machine Recognition of Faces: A Survey, Proc. IEEE, 1995](https://engineering.purdue.edu/~ece624/papers/challapa_facerecognition.pdf):
    - 1970s - Pattern recognition with simple features
    - 1980s - largely dormant…
    - 1990s - Karhunen-Loeve Tx, SVD (both with unsupervised learning), NN
    - late 1990s to 2000s - PCA, OCA, LDA, kernel methods, AAM, Boosting
    - current - deep CNNs, large datasets

#### Eigenfaces [Kirby…; Turk & Pentland, 1991](http://www.face-rec.org/algorithms/pca/jcn.pdf)
    - compute a basis set of faces, represent faces as weighted combination of basis faces
    - So instead of manually specifying the length of the nose, etc., the set of weights representing a face does the same automatically
    - Now that faces can be represented as a vector of weights, one can apply standard classification algorithms like Neight Neighbours, etc.

#### Local Binary Patterns [LBPs] [Ahonen et al., Face description with LBPs: Application to face recognition, TPAMI, 2006](http://www.ee.oulu.fi/mvg/files/pdf/pdf_730.pdf)
    - Make 256 binary patterns (patterns thresholded to make binary), run on image, make histogram of the number of each of the 256 patterns within the image
    - Image is divided into grids, and histograms are computed for each grid, and fused
    - Then use SVM, etc., to classify

- Other methods use the same sequence but using SIFT, SURF, etc. features

### Face Identity Verification
    - Check if two faces are of the same person or not, doesn’t matter what the name of the person is
    - Applications: image retrieval (indexing large video collections), clustering (visualization, clustering), to maintain privacy
    - Challenges: large amount of data, we need ~40TB of RAM to do this at world scale

#### Distance Metric Learning
    - To extract distance as a measure of similarity in semantics, we need to train distance metrics
    - Use Mahalanobis-like distance: (x_i - x_j)^T * M * (x_i - x_j)
    - Different supervision methods: class supervision, pairwise friend/foe supervision (hard) relative triplet constraints (soft)
    - Discriminative and Dimension-reducing embedding: M = L^T * L; distance = (x_i - x_j)^T * M * (x_i - x_j) = (x_i - x_j)^T * L^T * L * (x_i - x_j) = ||(L*x_i - L*x_j)||_2

- But most of these assume linear embedding

#### Non-linear Embeddings
    - [Taigman et al., DeepFace, CVPR 2014](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf): Use Deep CNNs as an embedding (instead of a linear embedding)
    - Proposed by Facebook, used Social Face Classification dataset (~4.4M images, ~4k identities) (Facebook proprietary)
    - After training, remove the final classification layer, use the last FC layer (after normalization) as features

#### Siamese Network
    - [S. Chopra et al., Learning a similarity metric discriminatively, with application to face recognition, CVPR 2005](http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf)
    - Use a siamese network to say if two images belong to the same person or not

- [Labeled Faces in the Wild (LFW) 2007 dataset](http://vis-www.cs.umass.edu/lfw/)
    - 13k images of faces of 4k celebrities
    - same, not-same pairs

- DeepFace Ensemble came close to human level verification (~97%) (should be taken with a grain of salt) (by Facebook, 2014)

#### VGG Face
    - By Oxford, in 2015
    - Semi-automatic creation of large publicly available dataset  - 2.6M images, 2.6k identities (weakly made verification, then human annotated, possibly in Hyderabad)
    - Used Triplet Loss, and an adaptive objective, achieved comparable results to DeepID and FaceNet

- But, the problem is not solved
    - Compression Loss: all images are being compressed to save memory
    - Large scale: in a large dataset, it is highly possible to find a different face with similar illumination/pose; without distractors 97% accuracy, with distractors 70% accuracy

### Age Estimation
    - [Liu et al., AgeNet: Deeply Learned Regressor and Classifier for Robust Apparent Age Estimation, CVPRW 2015](www.jdl.ac.cn/doc/2011/201611814324881700_2015_iccvw_agenet.pdf)
    - Task is to find the human estimation of _apparent_ age, not the real age
    - Pre-train a multi-class Face Classification network -> Fine-tune with Real Age from, say, passport data -> Fine-tune with Apparent Age

### ACTION

- Datasets:
    - Action in videos: [KTH Dataset (2004)](http://www.nada.kth.se/cvap/actions/)
    - [Blank et al., Actions as space-time shapes, ICCV 2005](http://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html)
    - Action in sports: [UCF Sports (2008)](http://crcv.ucf.edu/data/UCF_Sports_Action.php)
    - [Youtube actions (2009)](http://www.cs.ucf.edu/~liujg/YouTube_Action_dataset.html)
    - Actions in videos: [Hollywood (2009)](http://www.di.ens.fr/~laptev/actions/hollywood2/)

- Bag of Words might not be good because - only edges 

### Spatio-Temporal Interest Points [Ivan Laptev, Space Time Interest Points, ICCV 2003](http://ftp.nada.kth.se/CVAP/users/laptev/iccv03.pdf)
    - Find interest points that change in both space and time
    - Use BoW on spatio-temporal interest points

 ### Dense trajectories: [Wang et al., Dense Trajectories…, IJCV 2013](https://hal.inria.fr/inria-00583818/document)
    - Use motion wisely to figure out trajectories (within 15-20 frames, mostly <1sec), use information around trajectories to suppress background trajectories (like camera motion)
    - Dense sampling at multiple scales -> Tracking in each scale -> Feature description of trajectories (using HoG, HoF, MBH)
    - These features play the same role as spatio-temporal interest points
    - Classification can be done based on these features
    - Can be used with other aggregative methods like Fischer encoding

### Two Stream CNNs [[Simoyan and Zisserman, NIPS 2014]()]
    - Use appearance + motion
    - Input video -> Single frame -> 1 ConvNet for spatial stream, Multi-frame optical flow -> 1 ConvNet for temporal stream -> fuse both streams at the end
    - Optical flow: [Brox et al., High accuracy optical flow estimation based on a theory for warping, ECCV 2004](http://www.mia.uni-saarland.de/Publications/brox-eccv04-of.pdf)

- Previous standard - iDT (improved Dense Trajectories)

- Then people thought convolution itself can be done in both spatial and temporal dimensions, so-

### [3D ConvNets [Tran et al., Learning Spatio-Temporal Features with 3D Convolutional Networks, ICCV 2015]](https://arxiv.org/abs/1412.0767)
    - 3D ConvNets as general video descriptors
    - Train on large datasets
    - C3DD + iDT + Linear SVM

### [Dynamic Image Networks [Bilen et al., CVPR 2016]](https://arxiv.org/abs/1612.00738)
    - To reduce actions into single images
    - Dynamic image = RGB image to summarize video = Appearance + Dynamics
    - Rank Pooling: pool frames from the video according to their rank, but it is not differentiable
    - Dynamic Images are more suitable to dynamic actions (push-ups, etc.), while RGB images are more suitable to static actions (playing the piano, etc.)

### [Kinetics Dataset](https://deepmind.com/research/open-source/open-source-datasets/kinetics/)
    - [Suleiman and Zisserman, the Kinetics Dataset](https://arxiv.org/abs/1705.06950)
    - [Carreira and Zisserman, Quo Vadis, Action Recognition? CVPR 2017](https://arxiv.org/abs/1705.07750)
    - The nail on the coffin
    - Data was the bottleneck - proposed Kinetics Dataset
    - Convert 2D ConvNets into 3D, pre-trained as 2D and repeated in time
    - Huge jump in UCF-101 (98%) and HMDB-51 (80%) datasets accuracy with pre-training on Kinetics dataset
