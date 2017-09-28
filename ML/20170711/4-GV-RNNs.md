# 20170711

## [Girish Varma](https://www.iiit.ac.in/people/faculty/girish.varma/) - RNNs (14:00 to 15:00)

### SCENE TEXT

- Use RNN to learn CRF

- Without character segmentation using RNN (Zisserman)

- https://arxiv.org/abs/1603.03101
    - Scene text with char-level language modelling
    - With attention modelling

### [Connectionist Temporal Classification (CTC)](http://www.cs.toronto.edu/~graves/icml_2006.pdf)

- Removes need for chal-level segmentation

- Classifies s***i***m**p**l**e as simple

- CTC Loss
    - Let B be the decoding function: B(s**i**m***p**l*e)  = simple
    - p(simple) = {\sum}_{w that decodes to simple}p(w)
    - Loss = 1 - p(simple)

- But there are too many words that can decode to a simple word

- CTC with Dynamic Programming

- CNN + RNN + CTC: CRNN

### HARD ATTENTION MODELLING

- [Recurrent models of visual attention](https://arxiv.org/abs/1406.6247)

- How to predict the window containing the next character? (in MNIST, say)

