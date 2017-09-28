# 20170710

## [Vineeth Balasubramanian](http://www.iith.ac.in/~vineethnb/) - Backprop and Gradient Descent (11:30 to 13:00)

- Loss function from $J$ and $W_{i, j}$

- Backprop

- Gradient Descent

-  Unfortunately backprop doesn’t work very well with many layers

- 40s: perceptrons; 50s: MLPs, apparently MLPs can solve any problem; 70s: MLPs cannot solve non-linear problems, like XOR; 90s: Hinton revived NN using non-linear activations and backprop

### Local minima
    - NN loss functions are non-convex
        - non-linear activations
        - NNs have multiple local minima
    - So, weight initialization determines which local minimum your deep learning-optimized weights fall into
        - Also, weights are much more higher-dimensional
    - Gradient descent: drop a blind person on the Himalayas, ask them to find the lowest point
    - Non-identifiability problem/symmetry problem: any initialization is fine, any local minimum is OK
    - “Almost all local minima are global minima”

### Saddle Points
    - Gradient=0 could also occur at saddle points!
    - Saddle Point = minimum in one dimension, maximum in another
    - As the number of dimensions increases, the occurrence of saddle points is also greater

- There is a community that does not believe in deep learning, because deep learning does not provide any guarantees. SVMs, etc. provide guarantees on finding an optimal solution. Such guarantees do not exist with deep learning

### Cliffs
    - Highly non-linear error surfaces may also have cliffs
    - Occur for RNNs

### Vanishing/Exploding gradient
    - Gradients cascaded back might not have enough value by the first layers to change them much
    - This is why we don’t want networks with too many layers: ResNet-1000 performed worse than ResNet-152
    - Just to solve this problem in RNNs, LSTMs were invented
    - Exploding gradients occur when gradients (maybe for activations other than sigmoid), are >1
    - Exploding gradients could potentially be taken care of by clipping gradients at a max value

- Slow Convergence - no guarantee that:
    - network will converge to a GOOD solution
    - convergence will be swift
    - convergence will occur at all

- Ill conditioning: High condition number (ratio of max Eigenvalue to min Eigenvalue)
    - If the condition number is high, surface is more elliptical
    - If condition number is low, surface is more spherical, which is nice for descending

- Inexact gradients, how to choose hyperparameters, etc.

### Batch GD, Stochastic GD, Mini-Batch SGD

- Batch GD: Compute the gradients for the entire training set, update the parameters
    - Not recommended, because slow

- Stochastic GD: Compute the gradients for 1 training example drawn randomly, update the parameters
    - Loss computed in Batch GD is the average of all losses from training examples
    - So Batch GD and SGD don’t converge to the same point
    - It’s not called Instance GD, it’s called Stochastic GD because the samples are drawn randomly. There are proofs of convergence given depending on the fact that samples are random.

- Mini-Batch SGD: Compute the gradients for a mini-batch of training examples drawn randomly, update the parameters

#### Advantages of SGD over Batch GD
    - Usually much faster than Batch GD
    - Noise can help! There are examples where SGD produced better results than Batch GD

#### Disadvantage of SGD
    - Can lead to no convergence! Can be controlled using learning rate though

### FIRST-ORDER OPTIMIZATION METHODS

### Momentum
    - Momentum == Previous Gradient
    - Can increase speed when the cost surface is highly non-spherical
    - Damps step sizes along directions of high curvature, yielding a larger effective learning rate along directions of low curvature
    - Can help in escaping saddle points… But usually need second order methods to escape saddles

### Accelerated Momentum

### Nesterov Momentum
    - Compute the Gradient Delta from loss at the weights updated by the previous gradient delta, then combine this Gradient Delta with the previous gradient delta
    - Somehow works better than just momentum
