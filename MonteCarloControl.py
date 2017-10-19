import sys
import Environment
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

np.random.seed(3456)

e_action = Environment.action
reward = Environment.reward
step = Environment.step
draw = Environment.draw
Qsa = {}
Nsa = {}
N_0 = 10


# State space
states = []
for d_c in range(1, 11):
    for p_s in range(1, 22):
        states.append((d_c, p_s))

# Initialize q values
for _ in states:
    Qsa[_, e_action['hit']] = 0.0
    Qsa[_, e_action['stick']] = 0.0

# Initialize N(s,a)
for _ in Qsa:
    Nsa[_] = 0


def get_action(state):

    e = N_0/(N_0 + min(Nsa[state, e_action['hit']], Nsa[state, e_action['stick']]))

    if random.random() < e:
        action = random.randint(0, 1)
    else:
        action = np.argmax(np.array([Qsa[state, e_action['hit']], Qsa[state, e_action['stick']]]))

    return action


def episode():

    # Set first state
    dealer_card = draw(Environment.my_black_deck)
    player_sum = draw(Environment.my_black_deck)

    visited = {}

    state = [dealer_card, player_sum]
    action = get_action((state[0], state[1]))

    while not all(i == 0 for i in state[0:2]):

        visited[(state[0], state[1]), action] = 0
        Nsa[(state[0], state[1]), action] += 1

        # Get next state
        next_state = step(*state[0:2], action)
        # Get next action
        next_action = get_action((state[0], state[1]))

        state = next_state
        action = next_action

    for _ in visited:
        Qsa[_] += (1 / Nsa[_]) * (state[2] - Qsa[_])

epochs = 1000000

print('\nStarting Monte Carlo Control... \n')
print('Running Episode:\n')

# Running experiment
for _ in range(epochs):

    episode()
    sys.stdout.write('\r  \r')
    sys.stdout.write(str(_ + 1))
    sys.stdout.flush()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d', )

ax.set_xlabel('Dealer card')
ax.set_ylabel('Player sum')
ax.set_zlabel('State-Value')

x, y, z = [], [], []

f = open("checkQ.txt", "w")

for _ in states:
    # Writing checkQ file
    for a in range(2):
        f.write(str(_) + ',' + str(a) + ',' + str(Qsa[(_, a)]) + '\n')

    y.append(_[1])
    x.append(_[0])
    # Computing V*
    state_value = max([Qsa[(_, 0)], Qsa[(_, 1)]])
    z.append(state_value)

f.close()
ax.azim = 230
ax.plot_trisurf(x, y, z, linewidth=.02, cmap=cm.jet)
plt.show()






