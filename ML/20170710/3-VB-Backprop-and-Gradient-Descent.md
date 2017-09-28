# 20170710

## [Vineeth Balasubramanian](http://www.iith.ac.in/~vineethnb/) - Backprop and Gradient Descent (13:30 to 15:30)

- There is an optimal learning rate with which one can reach the global minimum in one step. It is equal to the inverse of the double derivative at the point from which we start

### RMSprop

### AdaGrad
   - Gradients diminish faster here… RMSprop takes care of this

### Adam
    - Similar to RMSprop with momentum

### AdaDelta
   - Works well in some settings

- Vanilla SGD may get stuck at saddle points

- Many papers use vanilla SGD without momentum, but best bet is Adam

- ![alt text](http://ruder.io/content/images/2016/09/contours_evaluation_optimizers.gif “Sebastian Ruder’s”)

### SECOND-ORDER OPTIMIZATION METHODS

- Gradient descent comes from approximating the Taylor series expansion to just the first order term

 - By considering the second-order terms:

$${\delta}w = -\frac{f’(x_{0}}{f’’(x_{0})}$$

- So we see that the gradient update is inversely proportional to the second derivative!

- So, gradient update = $-H^{-1} * g$, where $H$ is the Hessian, $g$ is the gradient

- The ideal learning rates are the Eigenvalues of the Hessian matrix

- But, the Hessian is not feasible to compute at every step, let alone its inverse and eigenvalues
    - Computational burden
    - Cost-performance tradeoff is not attractive
    - There are some Newton’s methods to try, but they are attracted towards saddle points

- So we use other approximations like the Quasi-Newton methods
    - We compute one Hessian, then we compute the difference to add to get the next Hessian
    - E.g. LGFBS
    - Approximations: Fischer Information, Diagonal approximation

### Conjugate Gradient Descent
    - Two vectors $x$ and $y$ are said to be conjugate w.r.t. A matrix A if $x^{T}Ay = 0$
        - Here, y was the previous gradient update, A is the current Hessian, x is the new gradient update
    - Compute the gradient, subtract some part of the previous gradient to get the conjugate component, compute the learning rate

### Hessian-free Optimization
    - Approximate the Hessian using the gradients

### Natural Gradient
    - Instead of using Euclidean distance to traverse along the error surface, travel along the surface of the manifold itself, the geodesic, the natural gradient
    - [Revisiting natural gradients for deep networks, Razvan, ICLR 2014](https://arxiv.org/abs/1301.3584)

- Generally, second-order methods are computationally intensive, first-order methods work just fine!

### RECENT ADVANCES

### Gradient Descent

- Trying to understand the error surfaces, escaping saddles, convergence of gradient-based methods

- [Deep Learning without Poor Local Minima, NIPS 2016](https://arxiv.org/abs/1605.07110)
    - Theoretically showed that converging to a local minimum is as good as global minimum

- [Entropy-SGD: Biasing Gradient Descent into Wide Valleys, ICLR 2017](https://arxiv.org/abs/1611.01838)
   - Try to push gradient descent towards flat minima, because it is those minima into which most networks now converge to anyway

### Escaping Saddle Points

- [Escaping from Saddle Points, COLT 2015](http://proceedings.mlr.press/v40/Ge15.html)
    - Add Gaussian noise to gradients before update

- Degenerate Saddles
    - We found that most networks actually converge to saddles, not even flat minima
    - Saddles may be good enough

### REGULARIZATION

- Difference between ML and Optimization is, ML tries to generalize, by designing an optimization problem first

- Theoretical ML: Empirical Risk Minimization (ERM). But, ERL can lead to overfitting

- Avoiding overfitting is Regularization

- It’s not important to get 0 training error, it’s more important to find the underlying general rule

### Early Stopping
    - Stop at an earlier iteration before training error goes to 0
    - Of course not a very good idea
    - Error-change criterion: if error hasn’t changed much within n epochs, stop
    - Weight-change criterion: if change in max pair-wise weight difference hasn’t changed much, stop
        - Use max, it might be that only one direction is learning while all others are not

### Weight Decay
    - In addition to prediction loss, also add a loss with the magnitude (2-norm) of the weights
    - Don’t give me a very complicated network
    - Using L1-norm weight decay gives sparse solutions, meaning many weights go to 0, which helps in compression

### DropOut
    - Does model averaging
    - In training phase: randomly drop a set of nodes in hidden layer, with probability p
    - In test phase: use all activations, but multiply them with p
        - This is equivalent to taking the geometric mean of all the models if you ran them separately
    - This is equivalent to using an ensemble
    - Dropping nodes helps nodes in not getting specialized
    - [Srivastava et al., JMLR 2014](http://www.jmlr.org/papers/volume15/srivastava14a.old/source/srivastava14a.pdf)

### Regularization through Gradient Noise
    - Simple idea: add noise to gradient
    - [Neelakantan et al., Adding gradient noise improves learning for very deep networks, 2015](https://arxiv.org/abs/1511.06807)

### DATA MANIPULATION METHODS

### Shuffling inputs
    - [Efficient backprop, LeCun](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

### [Curriculum Learning [Bengio et al., ICML 2009]](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf)
    - Old idea, proposed by Elman in 1993
    - Use simpler training examples first

### Data Augmentation
    - Data jittering, rotations, color changes, noise injection, mirroring
    - [Deep image: Scaling up image recognition [Wu et al., 2015]](https://arxiv.org/abs/1501.02876)

### Data Transformation
    - Normalize/standardize the inputs to have 0 mean and 1 variance (refer to [Efficient Backprop paper](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf))
    - Decorrelate the inputs (probably using PCA)

### Batch Normalization
    - DropOut fell out of favour after this was introduced
    - Introduced Internal Covariance Shift
    - Normalize the activation of a neuron across its activations for the whole mini-batch; then multiply with a variance and add a shift, and learn those mean and variance parameters in training
    - BatchNorm eliminated the need for good weight initialization
    - Also helps in regularization, by manipulating parameters with the statistics of the mini-batch

### ACTIVATION FUNCTIONS

- Sigmoid: Historically awesome, but gradients fall to 0 if activations fall beyond a small range

- tanh: Same problem

- ReLU: created to counter the zero-gradient problem
    - But, ReLU is not differentiable
    - So we assume ReLU’s gradient as a sub-gradient: we define the gradient at 0
    - Dying ReLU problem: If even once the activation goes to negative, the ReLU does not let any neuron before it learn anything during backprop

- LeakyReLU: created to counter the Dying ReLU Problem

- Parametric ReLU: Same as LeakyReLU, but a different slope for negative values than LeakyReLU

- ELU
    - ReLU gives a positive value as the average, but they said the average is supposed to be 0
    - So they made ELU which maintains the average 0
    - For negative activations: ${\alpha}*({\exp}^{r} - 1)$

- MaxOut: generalization of ReLU
    - Take the max of a bunch of network activations

- ELUs are slowly becoming popular

- Never use sigmoid

- Try tanh, but expect it to work worse than ReLU/MaxOut

### LABELS

- Try smooth labels (0.9 instead of 1)

### LOSS FUNCTIONS

- Cross-entropy (will simplify to similar to MSE)

- Negative Log-Likelihood (will simplify to similar to CE)

- Classification Losses, Regression Losses, Similarity Losses

### WEIGHT INITIALIZATION

- Don’t initialize with 0, no backpropagation, no learning

- Don’t initialize completely randomly, non-homogeneous networks are bad

- Most recommended: [Xavier Init [Xavier Glorot, and Yoshua Bengio, Understanding the difficulty of training deep networks, AISTATS 2010]](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
    - Uses fan_in + fan_out info to draw samples from a uniform distribution

- Caffe made a simpler implementation

- He made a better implementation because Caffe’s didn’t work for ReLU: [Delving deep into rectifiers, [He et al.]](https://arxiv.org/abs/1502.01852)
