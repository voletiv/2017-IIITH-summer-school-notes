# 20170703

## [Girish Verma](https://www.iiit.ac.in/people/faculty/girish.varma/) - AlexNet and Beyond (13:30 to 15:30)

- [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) (ImageNet 2012 Winner)

![alt text](https://kratzert.github.io/images/finetune_alexnet/alexnet.png “AlexNet”)

### [Dropout (Srivastava et al., 2015)](http://www.jmlr.org/papers/volume15/srivastava14a.old/source/srivastava14a.pdf)

![alt text](http://everglory99.github.io/Intro_DL_TCC/intro_dl_images/dropout1.png “Dropout”)

    - In each iteration, randomly choose some weights to zero out their outputs
    - Train the network
    - At testing time, use all neurons, don’t zero
    - So, each neuron does not depend a lot on other neurons, we are eliminating such dependencies
    - Obviously, it takes more number of epochs to achieve the same training accuracy as that without Dropout, but the testing accuracy increases very well

- Dropout was used in AlexNet

### [ZFNet](https://www.cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf) (ImageNet 2013 Winner)

![alt text](https://adeshpande3.github.io/assets/zfnet.png "ZF Net")

    - Looks very similar to AlexNet, but they tried to interpret the feature activity in the intermediate layers
    - They visualized the outputs of each layer
    - ZFNet was an improvement on AlexNet by tweaking the architecture hyperparameters, in particular by expanding the size of the middle convolutional layers
    - Also, they did Input Jittering
    - #### Input Jittering
        - Scale down the largest dimension to 256
        - Consider 5 images as sub-images of this 256-sized image of 224x224 size
        - Flip the images to make a total of 10 images
        - Use all of these to train the network

### [OverFeat (Pierre Sermanet, …, LeCun)](https://arxiv.org/abs/1312.6229)
    - Winner of ImageNet localization task 2013
    - Training to classify, locate and detect objects improves accuracy of all three
    - No need to jitter, takes care of cropping and scaling within the network itself

### [GoogLeNet/Inception](https://arxiv.org/abs/1409.4842) (runner up in ImageNet 2014)

![alt text](http://redcatlabs.com/2016-07-30_FifthElephant-DeepLearning-Workshop/img/googlenet-arch_1228x573.jpg “GoogLeNet/Inception”)

    - They simply doubled the number of hidden layers to 22
    - They saw that the FC layers contain the most number of parameters. So they reduced the number of parameters in the FC layer to the bare minimum, instead compensating via the convolutional layers
    - They carefully designed convolutional layers called Inception modules
    - #### Inception Modules

![alt text](https://cpmajgaard.com/blog/assets/images/parking/inception.jpg “Inception module”)

        - Each inception module is performing 1x1, 3x3, 5x5 and 3x3 with max pooling simultaneously
    - Stack Inception modules together

### [VGGNet](https://arxiv.org/abs/1409.1556) (winner of ImageNet 2014)
    - Deeper is better philosophy: multiple 3x3 filter layers have the same effect as 5x5 or 7x7 or bigger filters
    - Uses 19 layers

### [ResNet](https://arxiv.org/abs/1512.03385) (winner of ImageNet 2015)

![alt text](https://qph.ec.quoracdn.net/main-qimg-cf89aa517e5b641dc8e41e7a57bafc2c “ResNet”)

    - Network which won the competition was 110 layers deep
    - Deep networks have vanishing gradient problem. ResNet overcomes this using Skip connections
    - Skip connections: simply add gradients from a much further layer with those from regular backprop

### Ensembles
    - Train multiple networks and take a majority vote

- Batch Normalization (from CS231n lecture slides)

### CLASSIFICATION + LOCALIZATION (single object)

- Classification: get the object label, Localization: get the bounding box of the object
    - Use a classification network to classify the object
    - Use another network to get the bounding box

### DETECTION

- Task: put a bounding box on every instance of any class

#### [R-CNN](https://arxiv.org/abs/1311.2524)

![alt text](https://qph.ec.quoracdn.net/main-qimg-c96241e4e90c2b8509c4b1e87965965a “R-CNN”)

    - Image -> Extract Region Proposals -> Compute CNN features for each of the proposed region -> Classify regions
    - But, many region proposals is a problem

#### [Fast R-CNN](https://arxiv.org/abs/1504.08083)
    - Operation can be optimized by passing full image through CNN, and then looking at the proposed region’s outputs from CNN
#### Semantic Segmentation
    - We can make the output layer have same dimensions as input image, and value/third dimension with class label
    - We need to unpool to get the same image size
    - Used in [DeconvNet](http://arxiv.org/abs/1505.04366) we convolve and then deconvolve to get the same image size as the output size

![alt text](http://cvlab.postech.ac.kr/research/deconvnet/images/overall.png “DeconvNet”)
    - [Good link](https://handong1587.github.io/deep_learning/2015/10/09/segmentation.html)! 

#### Instance Segmentation
    - Each instance of a class is labeled separately

### Sequence-to-Sequence problems

#### Recurrent Neural Network
    - Convert x_1, x_2, … , x_s to o_1, o_2, … , o_t
    - RNNs are designed to remember long-term dependencies

- SceneText (OCR problem): RNN + CNN

- Image and Video captioning - [Venugopalan et al., “Sequence to Sequence - Video to Text”, 2015](https://arxiv.org/abs/1505.00487)

- #### Visual Question Answering
    - Pass image through CNN, question through RNN, then point-wise multiply the two outputs and pass through a fully-connected network
    - [Keras implementation](https://github.com/anantzoid/VQA-Keras-Visual-Question-Answering)

- [CS231n lectures](http://cs231n.stanford.edu/syllabus.html)
