import itertools
import Environment
import random
import numpy as np
import matplotlib.pyplot as plt

e_action = Environment.action
reward = Environment.reward
step = Environment.step
draw = Environment.draw
Qsa = {}
phi = {}
dealer_intervals = [(1, 4), (4, 7), (7, 10)]
player_intervals = [(1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21)]

np.random.seed(321)

# State space
states = [(0, 0)]
for d_c in range(1, 11):
    for p_s in range(1, 22):
        states.append((d_c, p_s))


def get_action(state):

    eg = 0.1

    if random.random() < eg:
        action = random.randint(0, 1)
    else:
        # Get best action according to latest theta
        q_hit = float(phi[(state[0], state[1]), e_action['hit']]*theta)
        q_stick = float(phi[(state[0], state[1]), e_action['stick']]*theta)
        action = np.argmax(np.array([q_hit, q_stick]))
    return action


def episode(lamda):

    # Initialize Eligibility Traces
    e = np.matrix(np.zeros((36, 1)))

    global theta

    # Draw black cards
    dealer_card = draw(Environment.my_black_deck)
    player_sum = draw(Environment.my_black_deck)

    state = [dealer_card, player_sum]
    action = get_action((state[0], state[1]))

    while not all(i == 0 for i in state[0:2]):

        # Take action A
        next_state = step(*state[0:2], action)

        # Choose A' from S' derived from Q with e-greedy
        next_action = get_action((next_state[0], next_state[1]))

        # Accumulating traces
        ones = np.where(phi[(state[0], state[1]), action].T == 1)
        e[ones] += 1.0

        # Q value of next state
        Qnext = float(phi[(next_state[0], next_state[1]), next_action] * theta)

        # Q value of current state
        Qstate = float(phi[(state[0], state[1]), action] * theta)

        # Update parameters
        delta = next_state[2] + Qnext - Qstate
        e = lamda * e + phi[(state[0], state[1]), action].T
        theta += 0.01 * delta * e

        state = next_state
        action = next_action

    # Get updated Q values at the end of each episode
    for _ in Qsa:
        Qsa[_] = float(phi[_] * theta)


# Run experiment and collect mse
lamdas = np.arange(0, 1.1, 0.1)
runs = 10000

# Read Q* from MC experiment
f = open("checkQ.txt", "r")
lines = f.readlines()
q_star = [0.0, 0.0]
for x in lines:
    q_star.append(float(x.split(',')[3].strip('\n')))
f.close()

mse_l0 = []
mse_l1 = []
mse = []

# Initialize q values
for _ in states:
    Qsa[_, e_action['hit']] = 0.0
    Qsa[_, e_action['stick']] = 0.0

# Feature space construction:
# c is the cartesian product across dealer_intervals, player_intervals and possible actions
c = list(itertools.product(dealer_intervals, player_intervals, e_action.values()))

# phi is a dictionary of binary feature vectors for each state-action pair
for sa in Qsa:
    phi[sa] = np.matrix([int(_[0][0] <= sa[0][0] <= _[0][1] and _[1][0] <= sa[0][1] <= _[1][1] and _[2] == sa[1]) for _ in c])

for l in lamdas:

    # Initialize parameter theta randomly
    theta = np.matrix(np.random.rand(36, 1))

    for r in range(runs):

        episode(l)

        if l == 0:
            q_l0 = [Qsa[_, a] for _ in states for a in range(2)]
            mse_l0.append(np.sum(np.square(np.asarray(q_l0) - np.asarray(q_star))) / 420)
        elif l == 1:
            q_l1 = [Qsa[_, a] for _ in states for a in range(2)]
            mse_l1.append(np.sum(np.square(np.asarray(q_l1) - np.asarray(q_star))) / 420)

    q = [Qsa[_, a] for _ in states for a in range(2)]
    mse_l = np.sum(np.square(np.asarray(q) - np.asarray(q_star))) / 420
    print("lambda :" + str(l), "mse :" + str(mse_l))
    mse.append(mse_l)


plt.plot(range(10000), mse_l0, '-r', label='lamda = 0')
plt.plot(range(10000), mse_l1, '-g', label='lamda = 1')
plt.legend(loc='upper right')
plt.ylabel('MSE')
plt.xlabel('# of episodes')
plt.title('Linear Gradient Descent Mean Squared Error vs # of episodes')
plt.show()
plt.plot(lamdas, mse)
plt.title('Linear Gradient Descent Mean Squared Error vs # of episodes')
plt.ylabel('MSE')
plt.xlabel('lamda')
plt.show()
