import random
from collections import defaultdict

import numpy as np
from sklearn.utils.extmath import cartesian

np.random.seed(3456)

suits = np.array([-1, -1, 1, 1, 1, 1])
ranks = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
reward = {'win': 1, 'draw': 0, 'loose': -1}
action = {'hit': 0, 'stick': 1}

# Decks for writing check files
# these are only for convenience
# in writing the files
expanded_deck = [card for card in cartesian([suits, ranks])]
expanded_black_deck = [card for card in cartesian([[1], ranks])]

# Decks for playing the game
my_deck = [np.prod(card) for card in expanded_deck]
my_black_deck = [np.prod(card) for card in expanded_black_deck]


# Randomly draw from deck
def draw(deck):

    return random.choice(deck)


# Step function implementation
def step(dealer_card, player_sum, player_action):

    if player_action == action['hit']:

        player_sum += draw(my_deck)

        if player_sum < 1 or player_sum > 21:

            return [0, 0, reward['loose']]

        else:

            return [dealer_card, player_sum, 0]

    elif player_action == action['stick']:

        while 1 <= dealer_card < 17:

            dealer_card += draw(my_deck)

        if dealer_card < 1 or dealer_card > 21:

            return [0, 0, reward['win']]

        elif dealer_card > player_sum:

            return [0, 0, reward['loose']]

        elif dealer_card < player_sum:

            return [0, 0, reward['win']]

        else:

            return [0, 0, reward['draw']]

    # Should never reach this point
    return ['nan', 'nan', 'nan']


def write_check_files():

    # Calling draw() 1000 times for checkDraw file

    calls = [','.join(map(str, draw(expanded_deck))) for _ in range(1000)]

    d = defaultdict(int)

    # Calculating frequencies

    for card in calls:
        d[card] += 1

    # Writing checkDraw file

    f = open("checkDraw.txt", "w")

    for k in range(1, 11):

        f.write(str(k) + '  ' + ' 1' + '  ' + str(d["1," + str(k)] / 1000.0) + '\n')
        f.write(str(k) + '  ' + '-1' + '  ' + str(d["-1," + str(k)] / 1000.0) + '\n')

    f.close()

    # Writing check step files
    step_files = np.array([[1, 1, 0], [1, 10, 0], [1, 18, 1], [10, 15, 1]])

    for file in step_files:

        outcomes = [','.join(map(str, step(*file))) for _ in range(1000)]
        d = defaultdict(int)

        # Calculating frequencies

        for outcome in outcomes:
            d[outcome] += 1

        sorted_keys = sorted(list(d.keys()), key=lambda x: list(map(int, x.split(',')[:2])))

        # Writing check files

        f = open("checkStepDealer" + str(file[0]) + "Player" + str(file[1]) + "Action" + str(file[2]) + ".txt", "w")

        for k in sorted_keys:

            f.write(str(k) + ', ' + str(d[k]/1000.0) + '\n')

        f.close()

    return 0


if __name__ == '__main__':

    write_check_files()
