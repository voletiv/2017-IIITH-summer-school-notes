# 20170713

## [Kaishuk Mitra](http://www.ee.iitm.ac.in/kmitra/), IIT Madras (14:30 - 15:30)

### AUTOREGRESSIVE MODELS

### [RIDE by Theis et al.](https://arxiv.org/pdf/1506.03478.pdf)

- Recurrent Image Density Estimator

- Structure: SLSTM + MCGSM

- Spatial LSTM: Kind of a 2D LSTM

- Mixture of Conditional Gaussian Scale Mixtures

### [Pixel RNN by Oord et al.](https://arxiv.org/abs/1601.06759)

- Improvisation over sLSTM

- Uses two different LSTM architectures: RowLSTM, Diagonal BiLSTM

### [Pixel CNN](https://arxiv.org/abs/1606.05328)

- Replace RNNs with CNNs

- Can’t see beyond a point, but much faster to train

### Conditional Pixel CNN

### [PixelCNN++](https://arxiv.org/abs/1701.05517)

- PixelCNN produces discrete distribution, this one produces continuous distribution

### DISCRIMINATIVE MODELS

- Image de-noising, image de-blurring, image super-resolution

- Con: there has to be a separate model each type of degradation

- Degraded image: Y = A*X + n

- Generative modelling: versatile and general

- MAP inference: arg max_X p(Y | X)*p(X)

- Traditional approach: Dictionary learning
    - [Video dictionary [Hitomi et al., 2011]](http://www.cs.toronto.edu/~kyros/courses/2530/papers/Lecture-10/Hitomi2011.pdf)
    - But, difficult to capture long-range dependencies

- GANs are out of the question, since they don’t model p(x) explicitly. MAP inference is not possible.

- VAE takes a lot of computation: they approximate p(x) with a lower bound

