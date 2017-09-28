# 20170703

## [C. V. Jawahar](https://faculty.iiit.ac.in/~jawahar/) - Intro & Background (09:30 to 11:00)

- Obj: to understand and learn the recent-ish advances in the CV scene in the world

- Last year summer school: basics of deep learning, touched CV; this year, 2 summer schools: one for CV; the other for Deep Learning (with some overlap)

- Typical day plan: Breakfast, Session 1, Break, Session 2, Lunch, Session 3a, Session 3b, Break, Demo/La/Tutorial, Lab/Practice & Quiz, Dinner

- CV Goal - To make computers _understand_ images and videos

- Scene classification (outdoor, lakeside...), Object Classification (is that a car, or...), Object Detection (where is the car, ...), Semantic Segmentation (in what all pixels is the car, ...), Pose Estimation (which direction is the car facing, ...)

- [Fei-Fei, Koch and Perona, What do we perceive in a glance of a real-world scene?, Journal of Vision 2007](http://authors.library.caltech.edu/11195/1/LIFjov07.pdf)

- Image -> Text: Image Annotation, Image Caption Generation, Image Description

- History of CV - face detection, ...

- Recognition: Classification (Instance recognition, Category recognition, Fine-grain classification), ... , Labels

- Challenges : Occlusions, truncations, scale/size, articulation, Inter-class similarity, Intra-class variation

- Variations in problems: Binary Classification, Multi-class, Multi-Label, Multi-output

- Feature extraction: I -> X, Classification: X -> Y; End-to-end: can we do I -> Y?

- [Caltech 101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) (2003): dataset for basic-level classification; objects from 101 classes; considered a toy dataset now; possibly gained high accuracies quickly because images were captured for the purpose of classification

- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) (2005-2012): 20 object classes, 22,591 images; multiple tasks: Classification, Detection, Segmentation; [van Gool, Zisserman, IJCV 2015](http://www.robots.ox.ac.uk/~vgg/publications/2015/Everingham15/)

- [ImageNet (ILSVRC)](http://www.image-net.org/) (2010): 1000 object classes, 14,197,122 images; Classification as Top-5, [Karpathy, Fei-Fei, IJCV 2015](https://arxiv.org/pdf/1409.0575.pdf)

- [COCO](http://mscoco.org/): harder than ImageNet; 80 object classes, 300,000 images; Describe Images, Human Keypoints

- More datasets...

- Classification evaluation: Overlapped area between true bounding box and predicted? Intersection? Intersection/Union?

- Basic Detection: Image -> features of every possible rectangle -> rectangle with max probability of class; Update: region proposal

- Evaluation metric: Average Precision (AP), Mean AP (Precision averaged over all thresholds of classification)

- Success of classification/detection: ML, data, computation (GPUs)

### HISTORY OF VISION: low-level features

    - Extremely low-level vision: filtering
    - Edges: Canny (1968), Sobel, Prewitt, ...
    - Textures: Viola-Jones (2001), ...
    - Histograms: [SIFT (Lowe, 1999)](https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/lowe_ijcv2004.pdf), Shape contexts (Malik, 2001), Spacial Pyramid Matching (Lazebnik, Schmid, Ponce, 2006)](http://www.vision.caltech.edu/Image_Datasets/Caltech101/cvpr06b_lana.pdf), [DPM based on HOG (Felzenswalb et al., 2010)](https://cs.brown.edu/~pff/papers/lsvm-pami.pdf)
    - Bag of Words: histograms happened in Text domain, so we brought them to images, like histograms of textures
    - Bag of Visual Words: histogram of predefined visual textures (Visual Words)
    - [Bag of Words (Zisserman, 2003)](http://www.robots.ox.ac.uk/~vgg/publications/papers/sivic03.pdf), SIFT (Lowe, 1999, [2004]((https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/lowe_ijcv2004.pdf))), [HOG+SVM (Dalal and Triggs, 2005)](http://www.csd.uwo.ca/~olga/Courses/Fall2009/9840/Papers/DalalTriggsCVPR05.pdf)

### HISTORY OF VISION: Mid-level features
    - Semantic segmentation

### HISTORY OF VISION: High-level features
    - Deep Learning: you can learn low-level to mid-level to high-level features automatically without manual intervention
    - One-Hot to Rich Representations: [word2vec](https://arxiv.org/abs/1301.3781) in text (Mikolov, 2013)
    - CBOW: Given a sequence of words, can you predict the missing middle word? Skip-gram: Given one word, can you predict the sequence of words before and after?
### SEGMENTATION
    - As Clustering: group similar pixels together (unsupervised), distance based on color and position; not really acceptable anymore for any decent segmentation
    - K-Means, [Normalized Graph Cut](https://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf) (Malik, 2006)
    - Graph Cut: Label Pixels as Background (Source) or Object (Sink), make that as a graph, cut the graph so that there is no path between Source and Sink; use MRF, etc. for making and cutting the graph
    - Graph Cut by Energy Minimization: pairwise constraint on pixel values
    - [Grab Cut (Rother et al., 2004)](https://cvg.ethz.ch/teaching/cvl/2012/grabcut-siggraph04.pdf) using Iterated Graph Cuts: User initialization, iterate: learn foreground, learn background; user initialization provides supervision
    - Superpixels: group pixels together, now apply same techniques by assigning labels to superpixels instead of pixels
    - Semantic Segmentation: Class Segmentation (where are persons?), Instance Segmentation (class: persons, segment boundaries of each person), Segmentation from expression ("wearing blue")
