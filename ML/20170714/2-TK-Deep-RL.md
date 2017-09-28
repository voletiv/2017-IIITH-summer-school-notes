# 20170714

## [Tejas Kulkarni](https://tejasdkulkarni.github.io/), Google Deep Mind - Deep RL - I (12:00 to 14:00)

- Fundamentals, Value-based Deep RL, Policy-based Deep RL, Temporal Abstractions in Deep RL, Frontiers

- Reinforcement Learning: a sequential decision-making framework to express and produce goal-directed behaviour

- Deep Learning: a representation-learning framework to re-present, interpolate and sometimes extrapolate raw data at multiple levels of spatio-temporal abstractions

- Deep RL: Simultaneously learn representations and actions towards goal-directed behaviour

- We use Deep Learning, but extend the loss function temporally

- State = g(x_t) given observations x_t. Here, g(.) denotes a deep neural network

- Deep Q Network (DQN): Predict Q using a neural network

- ATARI: 84x84 screen, 18 discrete actions

- But, introducing a neural network made the problem divergent
    - So, make a copy of the network called Target Network, and update it every few episodes

- Also, game score ranges are not same across games
    - So, clip the scores at +1 and -1 for robust gradients.
    - But the system loses the ability to differentiate between arbitrary reward values!

- Experience Replay:
    - Sample correlations can cause divergence during optimization
    - Alleviate this issue by storing samples in a large circular buffer called Replay Buffer
    - [Mnih et al., Human-level control through deep reinforcement learning]()


- Deep Successor Q Network [Deep Successor Reinforcement Learning, Kulkarni, Saeedi, et al.](https://arxiv.org/abs/1606.02396)

- Deep Actor-Critic Algorithms

- Variance reduction in Policy Gradients

- Asynchronous Advantage Actor-Critic (A3C)

- [Adding Auxiliary Losses to A3C [Reinforcement Learning with unsupervised auxiliary tasks, Jaderberg et al.]](https://arxiv.org/abs/1611.05397)

- Temporal Abstractions
    - [Options framework [Sutton, Precup, and Singh, Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning](http://www-anw.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf)
     - Have deep values as well

- Hierarchical DQN
