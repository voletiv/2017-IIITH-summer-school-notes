# 20170711

## [Ankush Gupta](www.robots.ox.ac.uk/~ankush/) (Oxford) - More on RNNs (13:30 to 15:00)

### ATTENTION MECHANISM

- Compare source and target hidden states
- Score the comparison between the hidden states of a source and a target node -> Do this for all encoder nodes with one target node (Make scores) -> Scale them and normalize w.r.t. Each other (Make Alignment Weights) -> Weighted Average

- [Bahdanau at al., 2015 (attention mechanism)](https://arxiv.org/abs/1409.0473)
    - Example of a well-written paper

### Text Image to Text String (OCR)
    - Recurrent Encoder-Decoder with Attention
    - Fully convolutional CNN -> Bi-directional LSTM (to capture context) -> Attention over B-LSTM to decode characters

- Attention Mask: can tell which part of the input corresponded with maximal output
    - [[Donahue et al., CVPR 2015]](https://arxiv.org/abs/1411.4389)

- http://distill.pub/

### CONCLUSION

- RNNs solve the problem of variable length input and output

- Solves knowledge of previous unit (by passing state)

- Can be trained end-to-end

- Finds alignment between input and outputs (through attention also)

