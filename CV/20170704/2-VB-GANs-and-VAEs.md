# 20170704

## [Vineeth Balasubramanian](http://www.iith.ac.in/~vineethnb/) (IIT Hyderabad) - GANs and VAEs (13:20 to 16:40)

### Introduction to Generative Models

- Recognizing objects is fine, a harder problem is to imagine!

- Eg. Handwriting generation, colorize b/w

- DIscriminative models: anything that tries to draw a boundary

- Generative models: HMM, etc. - Assume data is generated from an underlying Gaussian distribution, learn the mean and variance of that Gaussian model

### GANs

- Developed by Ian Goodfellow in 2014: https://arxiv.org/abs/1406.2661

- Generator: Noise -> Sample

- Discriminator: Binary classifier - is input data REAL or FAKE?

- [DCGAN (Radford et al, 2015)](https://arxiv.org/abs/1511.06434) uses CNNs as Generator and Discriminator

- Generator and Discriminator play against each other, the Generator trying to cheat the Discriminator, and the Discriminator getting better at catching the Generator

- Generator G tries to:
    a) maximize D(G(z)), or minimize (1 - D(G(z)))
    b) minimize D(x)
    - Combine a) and b): minimize D(x) + (1 - D(G(z)))

- Discriminator D tries to:
    a) maximize D(x)
    b) minimize D(G(z))
    - Combine a) and b): maximize D(x) + (1 - D(G(z)))

- So, min_G max_D [D(x) + (1 - D(G(z)))] (Same equation!)

- Loss in a GAN never comes down, it keeps oscillating. Because there are two players playing against each other. 

- We don’t have a good metric to know when to stop training. If it looks good to your eyes, it’s probably time to stop.

- Pseudocode from https://arxiv.org/abs/1406.2661

- Maximizing likelihood = Minimizing MSE

- Illustration of G imitating the real distribution

- Pitfalls of GAN: No indicator when to finish training, Oscillation, Mode collapsing

#### [Hacks of DCGANs](https://github.com/soumith/ganhacks) by Soumith Chintala
    - Normalize image b/w -1 and 1
    - Use Tanh
    - Don’t sample from uniform, sample from Gaussian (spherical)
    - Use BatchNorm. If BatchNorm is not an option, use InstanceNorm.
    - Avoid sparse gradients
        - Stability of GAN suffers
        - Use LeakyReLU instead
    - Use soft/noisy labels (0.7-1.2 instead of 1, 0.0-0.2 instead of 0)
    - Occasionally flip the labels given to the discriminator
    - Use SGD for D, and Adam for G

#### Variations of GANs
    - [Vanilla GAN [Radford et al., 2014]](https://arxiv.org/abs/1406.2661)
    - [Conditional GAN [Mirza and Osindera, 2014]](https://arxiv.org/abs/1411.1784): Give the class label also to D and G. Perhaps avoids mode collapsing.
    - [Bidirectional GAN [Donahue et al., 2016]](https://arxiv.org/abs/1605.09782)
    - [Semi-supervised GAN [Salimans et al, 2016]](https://arxiv.org/abs/1606.03498): Give class label to D while training, get the class label of the input image from D, in addition to REAL or FAKE
    - [Info GAN [Chen et al., 2016]](https://arxiv.org/abs/1606.03657): Give class label only to G, get class label as well from D; can generate 3D faces with varying Azimuth (pose), Elevation, Lighting, Wide/Narrow
    - [Auxillary GAN [Odena et al., 2016]](https://arxiv.org/abs/1610.09585)

####  [Vector Space Arithmetic using DCGAN [Radford, Mets, Chintala., 2016]](https://arxiv.org/abs/1511.06434)
    - <Man with glasses> - <Man without glasses> + <Woman without glasses> = <Woman with glasses>
    - What’s interesting is we have mapped images to a vector space that is continuous and awesome enough to be able to do such vector operations

#### [Super-Resolution (SRGAN)](https://arxiv.org/abs/1609.04802)
    - Super-resolved blurred images
    - Introduced Perceptual Loss = Content Loss + Adversarial Loss

#### [iGAN: Interactive GAN](https://github.com/junyanz/iGAN)
    - Updates manifold with user interaction
    - Adds a loss with the manifold, in addition to Adversarial Loss

#### [VideoGAN [Torralba et al., NIPS 2016]](http://carlvondrick.com/tinyvideo/)
    - Noise Vector -> Deconvolution to generate video for Foreground, to generate image for Background -> Combine foreground and background to make video

- [Text2Image](https://arxiv.org/abs/1615.05396)

- [Shape Modelling out of 3D GANs](http://3dgan.csail.mit.edu) came out of CSAIL MIT

- Image-to-Image Translation: [Pix2Pix](https://phillipi.github.io/pix2pix), with interactive demo! [aXiv](https://arxiv.org/abs/1611.07004)

- [Cycle GANs](https://hardikbansal.github.io/CycleGANBlog), [arXiv](https://arxiv.org/abs/1703.10593):
    - Unpaired Image-to-Image Translation
    - In addition to regular GAN stuff, try to generate back the real image from the generated image

#### Resources for GANs:
    - [List of GANs](https://github.com/nightrome/really-awesome-gan)
    - [GAN zoo](https://github.com/hindupuravinash/the-gan-zoo)
    - Ian Goodfellow

Qs: Generated images from uniform range? Why use Gaussian?

### VAEs

- Autoencoder is a network that tries to predict its input itself

- Use is, the hidden layer can be smaller than the input, so a smaller representation can be learnt

- If input is completely random, then this compression is very difficult

- Denoising Autoencoder: during training, willfully add noise, and denoise it using the autoencoder.

- MANIFOLD
    - A low-level representation of a higher-dimensional data
    - ML experts try to find the lower-dimensional representation of just about everything

- [Deep Autoencoders - Salakhutdinov and Hinton, Science, 2006](https://www.cs.toronto.edu/~hinton/science.pdf) is the paper that revolutionized Deep Learning.
    - People used to use RBMs with pre-training as an initialization
    - That’s when deep networks got noticed
    - Now, pre-training is redundant because we now use glorot init and xavier init

- Probabilistic Graphical Models: P(x, z) = P(z)*P(X|Z), if X is in the next layer of a network with input Z

- VAE loss function: KL Divergence + Reconstruction Loss

- Reparametrization trick - introduced in 2014 - so VAE becomes back-propagatable

- Attention in Deep Learning for Vision: RNNs for captioning - [Show, Attend and Tell](https://arxiv.org/abs/1502.03044), CS231n lectures

- [(DRAW) Deep Recurrent Attentive Writer](https://arxiv.org/abs/1502.04623): generates images in phases; [youtube](https://www.youtube.com/watch?v=Zt-7MI9eKEo)

- Attention Mechanism: [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025), [youtube](https://www.youtube.com/watch?v=6NOQC_fl1hQ)
    - Tx the squashed MNIST digit into a more regular form using the STN, then input that into any MNIST digit recognizer

- [Sync-DRAW with Captions](https://arxiv.org/abs/1611.10314) - currently accepted at ACM

- [PhysNet](https://arxiv.org/pdf/1603.01312.pdf) - try to learn physical laws from images and predict them

vineethnb@iith.ac.in