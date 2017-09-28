# 20170712

## [Vinay Namboodiri](https://www.cse.iitk.ac.in/users/vinaypn/), IIT Kanpur - GANs - II (13:30 to 14:30)

### [Unrolled GANs](https://arxiv.org/abs/1611.02163)

- Based on letting the generator “peek ahead” at how the discriminator would see

- Does much better than DCGANs

### STABILITY/CONVERGENCE PROBLEMS

### [Wasserstein GAN](https://arxiv.org/abs/1701.07875), [GitHub](https://github.com/martinarjovsky/WassersteinGAN)

- Handles stability/convergence problems by letting the generator look ahead better

- Uses Earth-mover’s Distance / Wasserstein Distance

- But this means gradients could explode, so Weight Clamping is to be used (not BCE)

### [Improved WGAN](https://arxiv.org/abs/1704.00028) (just 2 months later)

- Improved resilience against exploding gradients: Gradient penalty if it deviates from unit norm

### INTERPRETABILITY PROBLEMS

### [InfoGAN](https://arxiv.org/abs/1606.03657)

- Motivation: GANs use z in an entangled way; if a representation is unentangled, it could be made more interpretable

- Maximises Mutual Information

- Variational Maximisation of Mutual Information
    - Use an approximate function Q(C | X) for p(C=c | X=x)

- Sharing the net between Q(C | X) and the discriminator D(x)

- MNIST example with 10-dimensional discrete class, and 2 continuous classes (possibly width adn rotation)

- Faces example: pose, lighting, elevation

### MODE COLLAPSE


### [Mode Regularized GAN [Che et al., 2016]](https://arxiv.org/abs/1612.02136)

1) Metric Regularizer
    - Trying to learn an autoencoder

2) Mode Regularizer

### MULTI-AGENT FRAMEWORKS

### [Generative Multi-Adversary Network [ICLR 2017]](https://openreview.net/pdf?id=Byk-VI9eg)

- Multiple discriminators

- Formidable Adversary: hold up against all discriminators
    - Can be thought of as boosting
    - Generator could give up when the adversaries get too formidable

- Forgiving Teacher: have a variety of soft discriminators that can combine the outputs of multiple discriminators

### [MAD GAN (Multi-Agent Diverse Generation)](https://arnabgho.github.io/MADGAN/)

### SEQUENTIAL GANs

### [Contextual RNN GAN for Abstract Reasoning Diagram Generation [AAAI 2017]](https://arnabgho.github.io/Contextual-RNN-GAN/)
