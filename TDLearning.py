import Environment
import random
import numpy as np
import pylab as plt

np.random.seed(4524)

e_action = Environment.action
reward = Environment.reward
step = Environment.step
draw = Environment.draw
Nsa = {}
Qsa = {}
N_0 = 10


# State space
states = [(0, 0)]
for d_c in range(1, 11):
    for p_s in range(1, 22):
        states.append((d_c, p_s))


def get_action(state):

    e = N_0/(N_0 + min(Nsa[state, e_action['hit']], Nsa[state, e_action['stick']]))

    if random.random() < e:
        action = random.randint(0, 1)
    else:
        action = np.argmax(np.array([Qsa[state, e_action['hit']], Qsa[state, e_action['stick']]]))

    return action


def episode(lamda):

    esa = {}

    # Initialize Eligibility Traces
    for _ in Qsa:
        esa[_] = 0.0

    # Draw black cards
    dealer_card = draw(Environment.my_black_deck)
    player_sum = draw(Environment.my_black_deck)

    state = [dealer_card, player_sum]
    action = get_action((state[0], state[1]))

    while not all(i == 0 for i in state[0:2]):

        Nsa[(state[0], state[1]), action] += 1

        # Take action A
        next_state = step(*state[0:2], action)

        # Choose A' from S' derived from Q with e-greedy
        next_action = get_action((next_state[0], next_state[1]))

        # Q value of next state
        Qnext = Qsa[(next_state[0], next_state[1]), next_action]

        # Q value of current state
        Qstate = Qsa[(state[0], state[1]), action]

        delta = next_state[2] + Qnext - Qstate

        # Accumulating traces
        # esa[(state[0], state[1]), action] = 1
        esa[(state[0], state[1]), action] += 1

        for _ in Nsa:
            # if Nsa[_] > 0:
            # Qsa[_] += (1 / Nsa[_]) * delta * esa[_]
            Qsa[_] += (1 / (1 + Nsa[_])) * delta * esa[_]
            esa[_] *= lamda

        state = next_state
        action = next_action

lamdas = np.arange(0, 1.1, 0.1)
runs = 10000

# Read Q*
f = open("checkQ.txt", "r")
lines = f.readlines()
q_star = [0.0, 0.0]
for x in lines:
    q_star.append(float(x.split(',')[3].strip('\n')))
f.close()

mse_l0 = []
mse_l1 = []
mse = []

# Run experiment
for l in lamdas:

    # Initialize q values
    for _ in states:
        Qsa[_, e_action['hit']] = 0.0
        Qsa[_, e_action['stick']] = 0.0

    # Initialize N(s,a)
    for _ in Qsa:
        Nsa[_] = 0

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
plt.title('Sarsa(l) Mean Squared Error vs # of episodes')
plt.show()
plt.plot(lamdas, mse)
plt.title('Sarsa(l) Mean Squared Error vs lambda')
plt.ylabel('MSE')
plt.xlabel('lamda')
plt.show()
