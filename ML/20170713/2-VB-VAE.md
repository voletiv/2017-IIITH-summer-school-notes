# 20170713

## [Vineeth Balasubramanian](http://www.iith.ac.in/~vineethnb/index.html) - VAE (11:00 to 12:30)

### VARIATIONAL AUTO-ENCODER

- Latent variable mapping [Aaron Courville, Deep Learning Summer School 2015]
    - Trying to discover the manifold in the higher-dimensional space, trying to find the lower-dimensional numbers

- Example of Face Pose in x axis and Expression in y axis from the Frey Face dataset

- We can sample from z and generate an x using a transformation G: x = G(z)

- How to get z, given the x’s in our dataset

- [Auto-Encoding Variational Bayes, Kingma and Welling, ICLR 2014]

- Estimate ${\theta}$ without access to latent states

- PRIOR: Assume Prior $p_{\theta}(z)$ is a unit Gaussian

- CONDITIONAL: Assume $p_{\theta}(x | z)$ is a diagonal Gaussian, predict mean and variance with neural net

- So, from a sampled z, estimate a mean and variance from which to sample an x

- From Bayes’ theorem,

$$ p_{\theta}(z | x) = {\frac{p_{\theta}(x | z) * p_{\theta}(z)}{p_{]theta}(x)}} $$

- Here, p_{\theta}(x | z) can be found from the decoder network,  p_{\theta}(z) is assumed to be a Gaussian, BUT $p_{]theta}(x)$ is intractable. 
    - Because, to find $p_{]theta}(x)$, we need to integrate over all x’s on all values of z

- So, $p_{\theta}(z | x)$ is very hard to find, since we don’t know $p_{]theta}(x)$

- Instead, we approximate this posterior with a new posterior called $q_{\phi}(z | x)$, and then try to minimize the KL divergence between $q_{\phi}(z | x)$ and $p_{\theta}(z | x)

- We get $q_{\phi}(z | x)$ from the encoder network.

- Data point x -> Encoder: mean, (diagonal) covariance of $q_{\phi}(z | x)$ -> Sample $z$ from $q_{\phi}(z | x)$ -> Decoder: mean, (diagonal) covariance of $p_{\theta}(x | z)$ -> Sample $hat{x}$ from $p_{\theta}(z | x)$

- Reconstruction loss for $hat{x}$, Regularization loss on prior  $p_{\theta}(x)

- Reparametrization trick
    - Because we’re sampling in between, it is not back-propagatable, since we need to differentiate through the sampling process

- [Laurent Dinh Vincent Dumoulin’s presentation](http://www.iro.umontreal.ca/~bengioy/cifar/NCAP2014-summerschool/slides/Laurent_dinh_cifar_presentation.pdf)

### Semi-Supervised VAEs
    - M1 - vanilla VAE: use z to generate x
    - M2 - use z and a class variable y to generate x
    - M2 - use some inputs with labels and some without
    - M1 + M2
        - shows dramatic improvement
    - Conditional generation using M2

### Conditional VAE
    - z does not split the different classes, instead class label is given as a separate input

### [Importance-Weighted VAE, Salakhutdinov’s group, ICLR 2016](https://arxiv.org/abs/1509.00519)

### De-noising VAE

- Added noise to input

- Took $q_{\phi}$ as a mixture of Gaussians

### Deep Convolutional Inverse Graphics Network

- Train by a subset of components of z with one type of variation
    - Vary lighting and only sample from a subset of components, etc.

### Adversarial Autoencoders

- Ask a Discriminator to make out the difference between a sample z from the Encoder, and a sample from the actual prior (Gaussian)

### Applications
    - Image and video generation
    - Super-resolution
    - Forecasting from static images
    - Inpainting
