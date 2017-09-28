# 20170713

## [Kaushik Mitra](http://www.ee.iitm.ac.in/kmitra/), IIT Madras - Variational Inference and Autoencoders (09:00 to 10:30)

### AUTOENCODERS

- Linear AE is same as PCA

- But fails:
    - Just memorizes the data, doesn’t understand the data
    - Overcomplete case: hidden dimension h has higher dimensions
    - Problem: identity mapping

- Sparsity as a regularizer

- De-noising autoencoder [Vincent et al.]

- Contractive Autoencoder
    - Learns tangent space of manifold
    - Minimize Reconstruction error + Analytic Contractive loss
    - Tries to model major variation in the data

- Can we generate samples from manifold?
    - We need the lower-dimensional distribution
    - Let’s assume it is Gaussian

### VARIATIONAL AUTOENCODER

- Decoder samples z from a latent space, and decodes into the mean and variance from which to sample an x

- Encode: x -> Encode to a mean and variance for z, Decode: Sample a z -> Decode to a mean and variance for each dimensional value of x

- Terms:
    - Posterior: $p(z | x)$
    - Likelihood: $p(x | z)$
    - Prior: $p(z)$
    - Marginal: ${\int}p(x, z) dz$

- But, the marginal is intractable, because z is too large a space to integrate over

- So, decoder distribution p(x | z) should be maximized over q(z | x): minimize the log likelihood $E_{q(z | x)}[log p(x | z)]$

- Also, q(z | x) should be close to the z prior p(z) (Gaussian): minimize the KL Divergence $KL[q(z | x) || p(z)]$

- Importance Sampling
    - But, intractable

- Jensen’s inequality

** Concave: Graph is above the chord

- Variational Inference
    - Variational lower bound for approximating marginal likelihood
    - Integration is now optimization: Reconstruction loss + Penalty

