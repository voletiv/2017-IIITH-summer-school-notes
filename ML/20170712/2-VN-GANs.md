# 20170712

## [Vinay Namboodiri](https://www.cse.iitk.ac.in/users/vinaypn/), IIT Kanpur - GANs - I (11:00 to 12:30)

- Problem to solve: unsupervised learning of (visual) representations

- Given access to a lot of data, we want to generate the data. Generation => We have understood the data.

- Initially, we wanted to learn some properties of data by trying to reconstruct the data using a deep network
    - Unlabelled Data -> Deep Network -> SGD Objective with Reconstruction Loss
    - But, this network simply formed the identity function w.r.t. the input data.
    - Didn’t at all work for data other than input data
    - Deep learning is a bad student who just memorises whatever is given to him, but doesn’t understand anything about the data

- Then, [Hinton & Salakhutdinov, in Science 2006](https://www.cs.toronto.edu/~hinton/science.pdf) made an autoencoder, where they put a bottleneck in the middle of the deep network
    - This worked well, the network learnt a compressed version of the input data
    - But it wasn’t there yet, it just learnt a fancy compression

- [Denoising Autoencoders [Vincent et a;., ICML 2008]](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)
    - Give purposefully corrupted data -> Autoencoder -> Reconstruct the original image without noise

- [Sparse Autoencoder: Building High-level Features Using Large Scale Unsupervised Learning, ICML 2012](https://arxiv.org/abs/1112.6209)
    - Used 16000 processors, 1 billion connections, 10M images to solve the problem of unsupervised learning once and for all
    - Worked mostly for cat videos. Google argued that that’s because YouTube is full of cat videos

- GANs work! How do we know? After all, Stacked Autoencoders also went only so far... Results!
    - [BEGAN: Boundary Equilibrium GAN](https://arxiv.org/abs/1703.10717): Uses WGAN + encoders

- GANs: Counterfeit game analogy

- GAN Architecture [Ian Goodfellow et al., 2014](https://arxiv.org/abs/1406.2661)
    - z is some random noise = a latent representation of the images
    - GAN is not trying to reconstruct the input data
    - GAN is trying to make a model such that it generates data that is indistinguishable from real data
    - Generator vs Discriminator
    - G and D must be trained with complementary backprop gradients

- [GAN Tutorial, NIPS 2016](https://arxiv.org/pdf/1701.00160.pdf)

- G tries to:
    - Maximise D(G(z)), or minimise (1 - D(G(z)))
    - Minimise D(x)
    - => Minimise D(x) + (1 - D(G(z)))

- D tries to:
    - Maximise D(x)
    - Minimise D(G(z)), or maximise (1 - D(G(z)))
    - => Maximise D(x) + (1 - D(G(z)))

- Thus, GAN objective function:

$$min_G max_D [D(x) + (1 - D(G(z)))]$$

- To analyze average behaviour, we take expectation:
$$ min_G max_D E[D(x)] + E[(1 - D(G(z)))]

- At equilibrium, G is able to generate real images, and D is thoroughly confused

- Minimax Game:

$$ J^{(D)} = - {\frac{1}{2}}E_{x~p_{data}} log D(x) - {\frac{1}{2}}E_{z} log (1 - D(G(z)) $$
$$ J^{(G)} =  - {\frac{1}{2}}E_{z} log D(G(z) $$

- But Goodfellow’s paper didn’t have good enough results… Next good progress was using DCGANs

- [Unsupervised Representation Learning with DCGAN, Radford et al., ICLR 2016](https://arxiv.org/abs/1511.06434)
    - Used a Deep Deconvolutional Network as the generator: Removed fc layers, changed activations, used BatchNorm
    - First realistic generation of images. Autoencoder, etc. produced blurry and erroneous images, this one made good sharp images
    - Also, latent vectors capture semantics as well, like <Man with glasses> - <Man without glasses> + <Woman without glasses> = <Woman with glasses>

- [GAN Hacks by Soumith Chintala](https://github.com/soumith/ganhacks)

- [Data-dependent initialization](https://arxiv.org/abs/1511.06856); GitHub: [magic_init](https://github.com/philkr/magic_init)
    - To change pre-trained weights according to new data format (-1 to 1 instead of 0 to 1, etc.)

### Applications:

- Image from Text

- 3D Shape Modelling using GANs

- Image Translation: Pix2Pix
    - But this uses pairs of images

- [CycleGAN](https://github.com/junyanz/CycleGAN)
    - Trains from 1 to generated 2, and back from 2 to 1 (hence, cycle)

- [Next Video Frame Prediction [Lotter et al., 2016]](https://arxiv.org/abs/1605.08104)
