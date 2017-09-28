# 20170707

## [Vinay Namboodiri](https://www.cse.iitk.ac.in/users/vinaypn/), IIT Kanpur - Domain Adaptation (11:00 to 12:30)

- Different types of learning: [Pang and Yang, TKDE 2010](http://doi.ieeecomputersociety.org/cms/Computer.org/dl/trans/tk/2010/10/figures/ttk20101013452.gif)

- Domain Adaptation:
    - If X consists of two “domains”, we assume that the conditional probability of X|y is the same among domains, meaning the class labels are the same for X belonging to either domain, but the marginal probability P(X) is different for different domains

- Meaning, it might be the same blue dress that we need to classify, but it might be a different perspective on the image

- This is different from spitting training and testing data, because there we can pretty much assume that asymptotically we shall be sampling from both domains, but here we do not have information about the marginal probability of the domain we haven’t trained on

- Let’s look at Pre-Deep Learning methods

### SHALLOW Domain Adaptation Methods

### Instance Re-weighting
    - Take the instances, change the weights attached to each instance
    - Maybe using Maximum Mean Discrepancy Loss
    - TrAdaBoost method

### Model Adaptation: Adaptive SVM
    - Slightly perturb the classifier to better fit the small target domain instances
    - Online re-weighting of classifier

- But the next one gained more popularity

### Feature Augmentation: Geodesic Map Kernels
    - [Geodesic Flow Kernel for Unsupervised Domain Adaptation [B. Gong et al., CVPR 2012]]()
    - Map the Geodesic Flow between the subspaces (using principal components) of the source data and the target data on the Grassman Manifold
    - But this method is pretty cumbersome

### Feature Transformation: Subspace Alignment
    - This directly aligns the source and target subspaces using a Transformation Matrix M
    - M is learned by minimizing the Bergmen divergence: F(M) = ||X_S * M - X_T||^{2}_{F}; M* = argmin_{M}(F(M))
    - Worked best, among the classical approaches
    - ICCV 2013

### Dictionary Learning
    - Learn a common subspace, a Shared Dictionary, that can minimize the distance between the source and target points
    - This dictionary is a Shared Discriminative Dictionary
    - Then use a Reconstruction Error-based classification
    - CVPR 2013

### DEEP Domain Adaptation Methods

### Fine Tuning
    - Freeze most layers, train the last couple of layers
    - But, we are assuming that we do have some supervision for the target domain within the source domain

- What if there is no supervision in target domain?

- We need to put an additional constraint about the closeness of the source and target domains

- We want to design an NN such that the means of the activations of the source domain instances and the target domain instances are close to each other

### Deep Adaptative Networks
    - Kernel Mean Matching: re-weighting the training points such that the means of the training and test points in a reproducing Kernel Hilbert Space (KHS) are close. How to do this using CNNs?
    - Loss = CNN loss + MMD Regularizer
        - Here, the MMD regularizer is the RKHS distance between the mean enbedding
    - Next paper: [Michael Jordan et al., 2015](https://arxiv.org/abs/1502.02791)

### Deep Unsupervised Domain Adaptation
    - Assume many labeled examples in source domain, not many in the target domain
    - [Unsupervised Domain Adaptation by Backpropagation, Ganin and Lempitsky, ICML 2015](https://arxiv.org/abs/1409.7495)
    - Network: Input -> Feature extractor -> Label predictor (Classifier)
    - Right now, source sample features are quite apart according to their class, but target samples are not; meaning target samples won’t be classified properly, while source samples would be classified very well
    - We want to extract features where both the source and target samples are mixed up, meaning the source and target features are indistinguishable, implying that classification of such features would be equally good/bad for both source and target samples
    - So we add another branch from the Feature Extractor to classify whether a sample is coming from the source or target, and we want to train it adversarially so that it is not able to differentiate between a source and a target sample, implying their features are mixed up in the feature space
    - Correct (according to class) mix-up shall simultaneously be taken care of by the Label Predictor branch
    - To train adversarially, back-propagate the negative of the gradients from the Domain Classifier branch

### Adversarial Discriminative Domain Adaptation
    - Use separate CNNs for source and target
    - Pre-train only Source CNN, adversarially train both, test with Target CNN
