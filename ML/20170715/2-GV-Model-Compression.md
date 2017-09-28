# 20170715

## [Girish Varma](https://www.iiit.ac.in/people/faculty/girish.varma/), IIIT Hyderabad - Model Compression (11:10 to 12:30)

- Reduce memory usage by:
    - Compressing matrices: using sparse matrices, quantization
    - Design architecture to have lesser memory

- Pruning:
    - Fine pruning: prune the weights
    - Coarse pruning: prune neurons and layers
    - Static pruning: pruning after training
    - Dynamic pruning: pruning during training

### WEIGHTS

### Weight Pruning

- Drop the weights below a threshold

- Can be stored in an optimized way of matrix is sparse

- Ensuring sparsity: use L1 regularizer
    - L1 (absolute value) regularizer has more slope near 0 than L2 (quadratic) regularizer, so it is more likely to push values to 0 than L2

- [DeepNeuron: Simplifying the Structure of Deep Neural Networks [Wei Pan, Hao Dong, Yike Guo]](https://arxiv.org/abs/1606.07326)
    - Regularize the weights, drop neurons with weights close to 0

- [Learning Neural Network Architectures using Backpropagation [Suraj Srinivas, R. Venkatesh Babu]](https://arxiv.org/abs/1511.05497)

### Quantization

- [Compressing Deep Convolutional Neural Networks using Vector Quantization, Lubomir Bourdev](https://arxiv.org/abs/1412.6115)
    - Binary quantization leads to 20% drop in top-5 accuracy in ILSVC
    - Reisdual quantization

- [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding, Song Han et al.](https://arxiv.org/abs/1510.00149)
    - Non-uniform weight quantization via k-means clustering
    - Pruning + Quantization
    - Won best paper award in ICLR 2016

- [XNOR-Net: Image Classification using Binary Convolutional Neural Networks, Ali Farhadi](https://arxiv.org/abs/1603.05279)
    - Really fast
    - Uses fixed-point representation

### ARCHITECTURE

- What about architectures with compressed representations?

### GoogLeNet

- First architecture with improved utilization of computing resources

- Inspired from Network in Network

- Used a cascading of Inception modules

- Used Global Average Pooling as a replacement for fc layers
    - >90% of parameters in AlexNet and VGG Net were in fc layers
    - fc layers were prone to overfitting
    - Global Average Pooling pools an entire filter channel into 1 average value. So last conv layer needs to have as many channels as the number of output classes

### Inception v2

- Remove 5x5 convolutions; instead, use only 1x1 and 3x3, and use 2 3x3 to replace the 5x5 convolution
    - This itself produced 28% reduction in size

- Spatial factorizing into asymmetric convolutions

### [Xception](https://arxiv.org/abs/1610.02357)
    - Similar to inception; in the inception module, has a common 1x1 convolution and then apply 3x3 convolutions on top.

### [SqueezeNet](https://arxiv.org/abs/1602.07360)
    - Applies all the compression techniques

### [Mobile Nets](https://arxiv.org/abs/1704.04861)
    - More depth-wise convolutions

### Student-Teacher Networks

- [Do deep networks really need to be deep?](https://arxiv.org/abs/1312.6184)

- [Distilling the knowledge in a neural network, Hinton, NIPS-W 2014](https://arxiv.org/abs/1503.02531)

- Train a smaller network (student) with the hidden outputs of a deeper network (teacher)!
    - MIMIC Model (Multiple Indicator, Multiple Cause)

- Teacher could have resolved errors in labels

### [FitNets, Bengio](https://arxiv.org/abs/1412.6550)
    - Try to minimize error between student and teacher at intermediate states as well!

### Fast Algorithms for Convolutional Neural Networks, Andrew Lavin, Scott Gray](https://arxiv.org/abs/1509.09308)
    - Uses Winograd’s minimal filtering algorithms to reduce number of architecture ops
