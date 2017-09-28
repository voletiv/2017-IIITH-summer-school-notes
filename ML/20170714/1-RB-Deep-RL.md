# 20170714

## [Ravindran Balaraman](http://www.cse.iitm.ac.in/~ravi/), IIT Madras - Deep RL - I (09:00 to 12:00)

- [Sutton and Barto’s book](http://people.inf.elte.hu/lorincz/Files/RL_2006/SuttonBook.pdf)

- RL examples

- [Arcade Games [Mnih et al., Nature 2015]](http://dx.doi.org/10.1038/nature14236)
    - Learn to play from the video, from scratch

- [Mastering the game of Go [Silver et al., 2016]](https://gogameguru.com/i/2016/03/deepmind-mastering-go.pdf)

- Alpha Go [Silver et al., Nature 2016]

### THEORY

### Markov Decision Process
    - MDP: M is the tuple: $<S, A, {\Phi}, P, R>$
    - $S$: set of states
    - $A$: set of actions
    - ${\Phi} \subseteq SxA$: set of admissible state-action pairs
    - $P: {\Phi}xS -> {0, 1}$: probability of transition
    - $R: {\Phi}xR$: expected reward
    - Policy: ${\pi}: S->A$ (can be stochastic)
    - Maximize total expected reward
    - Learn an optimal policy

- Checkerboard/grid, with some squares blacked out, need to reach a corner (Goal) from one corner in the shortest possible path
    - Reward cannot be distance from goal, since that defeats the purpose of reinforcement learning
    - Reward can be -1 for every step taken

- Goal must be outside the agent’s direct control, thus outside the agent

- Agent must be able to measure success: explicitly, frequently during its lifespan

- We want to maximize the expected return, E[R_t]

- Continuous tasks: R_t = r_0 + r_1 + …

- Discounted tasks: R_t = r_{t+1} + {\gammma}r_{t+2} + {\gammma}^2t_{t+3} + … 
   - {\gammma} -> 0 => short-sighted, {\gammma}->1 => far-sighted

- Value function: Expectation of total return

- Bellman Equation for policy {\pi}:

$$ 
R_t = r_{t+1} + {\gammma}r_{t+2} + {\gammma}^2r_{t+3} + …
       = r_{t+1} + {\gammma}R_{t+1}            
$$

So,

$$
V^{\pi}(s) = E_{\pi}[R_t | s_t = s]
                = E_{\pi}[r_{t+1} + {\gammma}V(s_{t+1}) | s_t = s]
$$

Or,

$$
V^{\pi}(s) = {\sum}_a {\pi}(s, a) {\sum}_{s’} P^{a}_{ss’}[R^a_{ss’} + {\gammma}V^{\pi}(s’)]
$$

- Using Action Value Q:

$Q^{\pi}(s, a)$ is the expected return from state s, given an action a. So,

$$
Q^{\pi}(s, a) = {\sum}_{s’} P^{a}_{ss’}[R^a_{ss’} + {\gammma}V^{\pi}(s’)]
$$

So,

$$
V^{\pi}(s) = {\sum}_a {\pi}(s, a) Q^{\pi}(s, a)
$$

- Optimal Value function: the estimated long-term reward that you would get starting from a state and behaving optimally

- Optimal Policy: a mapping from states to actions such that no other policy has a higher long-term reward

- Bellman Optimality Equation for V*:

The value of a state under an optimal policy must equal the expected return for the best action for that state.

$$
V*(s) = max_{a{\in}A(s)} {\sum}_{s’} P^a_{ss’}[R^a_{ss’} + {\gammma}V*(s’)]
Q*(s) = max_{a{\in}A(s)} {\sum}_{s’} P^a_{ss’}[R^a_{ss’} + {\gammma}max_{a’}Q*(s’, a’)]
$$

V* and Q* are the solutions to this system of non-linear equations.

- One-step Q-learning:
$$
Q(s_t, a_t) <- Q(s_t, a_t) + {\alpha}[r_{t+1} + {\gammma}max_a Q(s_{t+1}, a) - Q(s_t, a_t)]
$$

The alpha-term is called Temporal Difference error (TD Error).

- [REINFORCE [Williams, 1992]](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)

- Simple Monte-Carlo Policy Gradient Algorithm

- Policy Gradient Theorem

- Use a parameterized representation for value functions and models
