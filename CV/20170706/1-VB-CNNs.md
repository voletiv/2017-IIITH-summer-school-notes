# 20170706

## [Vineeth Balasubramanian](http://www.iith.ac.in/~vineethnb/) - Visualizing, Understanding and Exploring CNNs (09:30 to 11:10)

- [UFLDL](http://deeplearning.stanford.edu/wiki/index.php/UFLDL_Tutorial): tutorials on deep learning

- Deep learning is awesome because, not much manual design of weights

- Understanding CNNs: visualize patches, visualize weights, etc.

### Visualize patches:
    -  Visualize patches that maximally activate neurons
    - what pattern/texture caused this particular neuron to fire?
    - [Rich feature hierarchies..., Malik et al., 2013](https://arxiv.org/abs/1311.2524)

### Visualize the weights: 
    - the weights are the filter kernels
    - Some look like Gabor filters (Gaussian over a sinusoid)

- Deep model compression - Winner of ICLR 2016 - they found that most of the weights are close to 0, meaning they are not important - [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](http://arxiv.org/abs/1510.00149)

### Visualize the representation space (e.g. with t-SNE)
    - t-SNE visualization [van der Maaten and Hinton, "Visualizing High-Dimensional Data Using t-SNE", Nov 2016, Journal of Machine Learning Research](http://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
    - t-SNE, IsoMaps… these are non-linear dimensionality reduction techniques

### Occlusion experiments:
    - Occlusion experiments, in [Visualizing and Understanding Convolutional Networks, Zeiler and Fergus, 2013](https://arxiv.org/abs/1311.2901)
    - They put grey boxes in random places in an image, forward passed it, and found in which cases was the right class predicted, so as to understand what in the original image caused the right classification

### Deconv approaches:
    - Start from a neuron of our choice to find out what it learns:
        1) Feed an image into a trained net
        2) Set the gradient of that neuron to 1, and the rest of the neurons in that layer to 0
        3) Backprop to image
    - Another way is Guided Backpropagation, from [Striving for Simplicity, by Dosovitskiy et al., 2015](https://arxiv.org/abs/1412.6806):
        - While backpropagating, don’t just set those neurons whose activations were negative (in case of ReLU) to 0, also set those gradients to 0 which are negative
        - This gives better visualization than just Deconv

### Optimization to Image

- [Deep Inside Convolutional Networks, Simonyan, Vedaldi, Zisserman, 2014](https://arxiv.org/abs/1312.6034)
    - How about taking dE/dX, where X is our input image, and finding out what is the X that minimises E?
    1) Set all gradients to 0
    2) Set the correct class neuron’s gradient to 1, back-propagate to the image
    3) Slightly change the image values…

- [Understanding Deep Image Representations by Inverting Them, Mahendran and Vedaldi, 2014](https://arxiv.org/abs/1412.0035)
    - Question: can we reconstruct what image looks similar to an input image to a network, based on the vector code at the output of that network for the input image?

- [DEEP DREAM](https://github.com/google/deepdream)

-  Image Style Transfer
    - Gatys et al.
    - deepart.io

### Fooling Neural Networks

- Question: can we use the above concepts of optimizing over the input image to maximize any class score, to fool a ConvNet?

- [Intriguing properties of Neural Networks, Szegedy et al., 2013](https://arxiv.org/abs/1312.6199)
    - Add a bit of noise to an image, the images start getting classified as “ostrich”

- [Deep Neural Networks are easily fooled](https://arxiv.org/abs/1412.1897) 
### Explaining CNNs: Class Activation Mapping

- After Conv layers, make a layer of the Global Average Pooled values of each channel, and make a classification layer with weights w1, w2, etc. corresponding to the GAP values
- Using the weights, compute w1*c1 + w2*c2 + ... , where c1, c2, … are the original channels, not the GAP values

- The resultant map is called Class Activation Map (CAM)

- This can tell us what area of the original image the classifier was focussing on to predict the right class

- Grad-CAM:
    - CAM required a re-training of a network. Guided Grad-CAM avoids this

- Guided Grad-CAM:
    - Use Grad-CAM with Guided Backpropagation

- Results of Guided Backpropagation, Grad-CAM, Guided Grad-CAM
