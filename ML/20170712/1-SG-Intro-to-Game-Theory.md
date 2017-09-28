# 20170712

## [Sujit Gujar](https://www.iiit.ac.in/people/faculty/sujit.g/), IIIT Hyderabad - Introduction to Game Theory (09:00 to 10:30)

- Game Theory: mathematical model of conflict

- Elements: Players, States, Actions, Knowledge, Outcomes, Payoff or Utility

- Assumptions:
    - Players are rational beings trying to maximize their payoff
    - All players have complete information of the game

- Pure Strategy: deterministic steps

- Prisoner’s Dilemma:

|                           | No Confess NC | Confess C |
| ------------------ | ------------------ | ------------ |
| No Confess NC |      -2, -2            |    -10, -1    |
|      Confess C    |      -1, -10           |    -5, -5     |

- Equilibrium: No player has any advantage deviating from it

- Playing (C, C) in the above Prisoner’s dilemma game is equilibrium.

- Some games have no equilibrium. Eg. Matching coins game:

|     |      H     |     T     |
| -- | -------- | -------- |
| H | 10, -10 | -10, 10 |
| T |  -10, 10 | 10, -10 |

    - In this case, players have to mix their strategies (of drawing H or T)

- Mixed strategy:
    - Playing actions (a_1, … , a_n) with probabilities (p_1, … , p_n)
    - Now, payoffs are not fixed, because strategies are randomized. We can only find Expected Payoff.
    - Mixed Strategy leads to Utility Theory by Neumann and Morgenstern

- In a Zero-Sum game, players follow a Min-Max Strategy

- GANs: Zero-Sum or Stackelberg Games
