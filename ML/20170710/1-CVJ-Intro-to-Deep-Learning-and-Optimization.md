# 20170710

## [C. V. Jawahar](https://faculty.iiit.ac.in/~jawahar/) - Intro to Deep Learning and Optimization (09:30 to 11:00)

- SVM

- Gradient Descent

- In addition to minimizing the reconstruction loss, we also want our model to be as simple as possible - so the magnitude of the weights must also be lesser

- PEGASOS: Primal Estimated sub-GrAdient SOlver for SVM

- Perceptron Learning: Perceptrons can only solve linearly-separable problems

- Deeper networks: MLP

- Backpropagation

### Motivations for Deep Architecture
    - Insufficient depth can hurt
    - Brain seems to to deep

- Learning representations, automating this

- History: Fukushima (1980), LeNet (1998), Many-layered MLP with BackProp (tried, but without much success and with vanishing gradients), Relatively recent work

### [ALEXNET, Krizhevskky et al, 2012](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - Winner of ImageNet 2012
    - Trained over 1.2M images using SGD with regularization
    - Deep architecture (60M parameters)
    - Optimized GPU implementation (cuda-convnet)
