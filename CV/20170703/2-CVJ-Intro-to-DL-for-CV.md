# 20170703

## [C. V. Jawahar](https://faculty.iiit.ac.in/~jawahar/) - Intro to DL for CV (11:30 to 12:45)

- Linear Classifiers

- Nearest neighbours

*Cholesky Decomposition

- Nearest neighbours for image annotation: "A new baseline for image annotation"

- SVM
    - Max-margin classifier
    - Key words: Training, testing, generalization, error, complexity (number of parameters), generative classifiers, discriminative classifiers

### [ALEXNET](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - After such classifiers came AlexNet which suddenly improved error from ~25% to ~15%
    - Appreciate what Alex did, it was very difficult to do the first time

- Over the years, deeper networks brought the ImageNet error rate down to human level

- LeNet (1989) - LeNet (1998) - AlexNet (2012) comparison

### Neural Networks

- MLP

- Back-propagate through the networks using gradient descent

- CNN: Locally connected networks with shared weights
    - FC (too many weights/parameters) -> Locally connected filters (many many weights, one set per, say, 3x3 filter going over the original image) -(BIG JUMP)> use shared weights to convolve over an image (much lesser number of parameters) => Convolutional layer with 1 feature map -> Convolutional layer with multiple feature maps
    - It is observed that the filters learned this way are similar to the filters we had manually tried to make many years ago
    - Pool: Shrink the output size by choosing only some of the outputs
    - Stride: Jump pixels to reduce number of parameters

- Activation functions: Pass the output of the CNN through a non-linearity to generalize

- Stack such layers together - ...=-ConvPoolNorm-ConvPoolNorm-...; this behaves similar to an MLP

- This is what Alex did. “[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)”

- CNN features are generic: Now, we can use the same network, remove the last classification layer, and use the features learnt till the penultimate layer to classify other object categories!
    - Train the CNN on a very large dataset like ImageNet
    - Reuse the CNN to solve smaller problems by removing the last (classification) layer

### Fine tuning
    - Extend to more classes (eg. from 1000 classes to another new 100 classes)
    - Extend to new tasks (eg. from object classification to scene classification) (Transfer Learning)
    - Extend to new datasets (eg. from ImageNet to PASCAL)

### Transfer Learning
    - People tried to see why same features could be used for different tasks
    - [“How transferable are features in deep neural networks?”, Bengio, NIPS 2014](https://arxiv.org/abs/1411.1792)

- Other popular deep architectures: Autoencoder, RBM, RNN, …

- Summary
