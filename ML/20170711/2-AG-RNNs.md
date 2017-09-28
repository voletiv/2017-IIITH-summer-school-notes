# 20170711

## [Ankush Gupta](www.robots.ox.ac.uk/~ankush/) (Oxford) - More on RNNs (11:00 to 12:30)

- Many flavours of Sequence-to-Sequence problems

- One-to-one (image classification), one-to-many (image captioning), many-to-one (video classification), asynchronous many-to-many (language translation), synchronous many-to-many (video labelling)

### RNN
- [input, previous hidden state] -> hidden state -> output

- RNNs model the joint probability over the sequences as a product of one-step-conditionals (via Chain Rule)

- Each RNN output models the one-step conditional $p(y_{t+1} | y_{1}, … , y_{t})$

### ENCODER-DECODER FRAMEWORK

- [Sutskever et al., 2014](); [Cho et al., 2014]()

- Can stack RNNs together, but in my experience any more than 2 is unnecessary

- Thang Luong’s Stanford CS224d lecture

- Loss function: Softmax + Cross-Entropy

- Objective is to maximize the joint probability, or minimize the negative log probability

- The encoder is usually initialized to zero

- If a long sequence is split across batches, the states are retained

### Scheduled Sampling
- During testing, it might just happen that the RNN gives one wrong output, and the error is compounded with time since the wrong output is fed as the next input!

- Scheduled Sampling is employed to take care of this

- During training, from time to time, sample from the output of the RNN itself and feed that to the next decoder input instead of the correct input

### REPRESENTATION: Feature Embeddings / One-Hot Encoding

#### Domain-specific features
    - ConvNet fc feature vectors for images
    - Word2Vec features for words
    - MFCC features for audio / speech
    - PHOC for OCR (image -> text string)

#### One-hot encoding

### Word-level
    - Usually a large lexicon of ~100k words
    - Cumbersome to handle
    - Softmax is unstable to train with such huge fan out number

- So we go for:

### Character-level
    - Represent as sequence of characters

### INFERENCE

- We don’t take argmax of the output probabilities because we will not optimize the joint probability then.

- Exact inference is intractable, since exponential number of paths with sequence length

- Why can’t we use Viterbi Decoder as in HMMs?***

### Beam Search with Approximate Inference

- So, we compromise with an Approximate Inference:
    - We do a Beam Search through the top-k output classes per iteration (k is usually ~20)
    - So, we start with the <start> token -> take the top-k output classes -> use each of them as the next input -> get the output class scores for each of the k potential sub-sequences -> sum the scores and take the top-k output classes -> use each of them as the next input …

### LANGUAGE MODELLING

- Use RNN so as to capture context

### SAMPLING/GENERATION

- Use “tau” as temperature to modify the output probabilities: s = s/tau
    - tau = 0 => prob is infinity for one word
    - tau = infinity => prob is flat, so you might not have trained your RNN at all

### WHAT HAS RNN LEARNT?

- Interpretable Cells
    - Quote-detection cell: one value of the hidden state is 0 when the ongoing sentence is within quotes, 0 else
    - Line length tracking cell: gets warmer with length of line
    - If statement cell
    - Quote/comment cell
    - Code depth cell (indentation)
