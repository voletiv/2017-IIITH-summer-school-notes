# 20170711

## [Girish Varma](https://www.iiit.ac.in/people/faculty/girish.varma/) - RNNs (09:30 to 10:30)

- RNNs

- Backpropagation through time

- Vanishing gradient problem

### GRUs

![alt text](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/10/Screen-Shot-2015-10-23-at-10.36.51-AM.png “WildML”)

- Use tanh instead of sigmoid

- Has two gates: an Update Gate, and a Reset Gate

- Update Gate:

$$z_{t} = {\sigma}(W^{(z)}x_{t} + U^{(z)}h_{t-1})$$

- Reset Gate:

$$r_{t} = {\sigma}(W^{(r)}x_{t} + U^{(r)}h_{t-1})$$

- New memory content, as a combination of new input and a fraction of old memory:

$$hh_{t} = tanh(Wx_{t} + r .* Uh_{t-1})$$

- Updated memory content, as a combination of fraction of old memory content and complementary new memory content:

$$h_{t} = z_{t} .* h_{t-1} + (1 - z_{t}) .* hh_{t}$$

- We can see that if z_{t} is close to 1, we can retain older information for longer, and avoid vanishing gradient.

### LSTMs

- LSTMs have 3 gates - Forget Gate, Input Gate, and Output Gate

![alt text](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-GRU.png] “Colah’s”)

### Bi-directional RNNs

### Stacking RNNs

