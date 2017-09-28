# 20170715

## [Mitesh Khapra](http://www.cse.iitm.ac.in/~miteshk/publications.html), IIT Madras - Deep Learning for Word Embeddings

- “You shall know a word by the company it keeps” - Firth., J. R. 1957:11

- In One-Hot Encoding, every pair of points is sq(2) Euclidean distance between them. Using any distance metric, every pair of words is equally distant from each other. But we want an embedding that captures the inherent contextual similarity between words

- A Co-occurrence Matrix is a terms x terms matrix which captures the number of times a term appears in the context of another term
    - The context is defined as a window of k words around the terms
    - This is also called a word x content matrix

- Stop words will have high frequency in the Co-occurrence Matrix without offering anything in terms of context. Remove them.

- Instead of counts, use Point-wise Mutual Information:
    - PMI(w, c) = log(p(c | w)/p(c)) = log(count(w, c)*N/(count(c)*count(w)))
    - So Mutual Information is low when both words occur quite frequently in the corpus but don’t appear together very frequently
    - PMI = 0 is a problem. So, only consider Positive PMI (PPMI): PPMI=PMI when PMI>0, PPMI=0 else

- It’s still very high-dimensional and sparse. Use PCA:
    - SVD: X_{mxn} = U_{mxk}{\Sigma}_{kxk}V^T_{kxn}, where k is the rank of the matrix X
    - Make k = 1, or any number lesser than the rank of X, and U*{\Sigma}*V^T is still an mxn matrix, but it is an approximation of the original X, wherein the vectors are projected along the most important dimensions, and it is no longer sparse

- X*X^T is the matrix of the cosine similarity between the words. X*X^T(i, j) captures the similarity between the i^{th} and j^{th} words. 

- But this is still high-dimensional. We want another approximation W, lesser dimensional than X, s.t. W*W^T gives me the same score as X*X^T
    - $X*X^T = (U{\Sigma}V^T)*(U{\Sigma}V^T)^T = (U{\Sigma}V^T)*{V{\Sigma}U^T} = U{\Sigma}*(U{\Sigma})^T$, because V is orthonormal (V*V^T = I).
    - So, U{\Sigma} is a good matrix to be our W, since it is low-dimensional (m x k).

- Iti pre-deep learning methods

### CONTINUOUS BAG OF WORDS (CBoW)

- Given a bag of n context words as the input to a neural network, predict the (n+1)^{th}word as the softmax output of the network.
