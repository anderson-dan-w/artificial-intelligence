
##pr3
prisoners_dilemma = [
 [( -5, -5), (-1,-10)],
 [(-10, -1), (-2, -2)]]

def payoff( game, player0, player1):
    return game[player0][player1]

solvable_0 = [[(3, 3), (6,5) ],
              [(1, 3), (2, 4)]]
solvable_1 = [[(4, 4), (9, 6), (9, 5)],
              [(8, 7), (8, 9), (8, 8)],
              [(7, 9), (7, 9), (7, 9)]]
solvable_2 = [[(7, 9), (9, 8), (4, 9), (7, 9)],
              [(8, 8), (9, 7), (4, 8), (8, 7)],
              [(9, 4), (9, 4), (5, 5), (8, 4)],
              [(8, 4), (9, 4), (4, 5), (8, 4)]]
not_solvable_0 = [[(3, 3), (6,5) ],
                  [(6, 4), (2, 3)]]
not_solvable_1 = [[(3, 1), (4, 1), (5, 1)],
                  [(3, 6), (2, 4), (3, 8)],
                  [(3, 3), (6, 1), (4, 4)]]
not_solvable_2 = [[(1, 8), (2, 7), (3, 8), (5, 9)],
                  [(2, 6), (3, 5), (4, 6), (1, 7)],
                  [(3, 7), (4, 4), (1, 5), (2, 6)],
                  [(4, 1), (1, 1), (2, 1), (3, 1)]]

def domination_compare(strat1, strat2):
    strat1Dominate = True
    strat2Dominate = True
    for play1, play2 in zip(strat1, strat2):
        if play1 <= play2:
            strat1Dominate = False
        elif play1 >= play2:
            strat2Dominate = False
    if strat1Dominate:
        return 1  # strat1 strictly dominates
    if strat2Dominate:
        return 2 # strat2 strictly dominates
    return 0     # neither strat strictly dominates

print(domination_compare((1, 3, 5), (2, 4, 6))) # should return 2
print(domination_compare((1, 3, 5), (0, 2, 4))) # should return 1
print(domination_compare((1, 3, 5), (0, 2, 6))) # should return 0

def strategyExtractor(game, player, strategy):
    if player != 0 and player != 1:
        print("Aren't these two player games? player must be 0 or 1 (This is Python, not MATLAB)")
        return []
    if player == 0:
        return [game[strategy][x][player] for x in range(len(game))]  # game is a square, so it's ok
    elif player == 1:
        return [game[x][strategy][player] for x in range(len(game))]
    print("How'd we get here? Programming error?")
    return [] # I think python might require a return? That can't be right; maybe I'm confusing it with C
              # where the gcc compiler I used wasn't quite introspective enough to realize that a return
              # inside the if, and then inside the else, meant all was covered. I think JAVA misses this
              # as well?

print(strategyExtractor(prisoners_dilemma, 0, 0)) # should return (-5, -1)
print(strategyExtractor(prisoners_dilemma, 1, 1)) # should return (-10, -2)

def strictlyDominated(strat1, strat2):
    for play1, play2 in zip(strat1, strat2):
        if play1 >= play2:
            return False # strat1 is NOT strictly dominated by strat2
    return True

def findDominatedStrats(game, player):
    dominated = []
    strats = [strategyExtractor(game, player, s) for s in range(len(game))]
    for i, strat1 in enumerate(strats):
        for j, strat2 in enumerate(strats):
            if i == j: # no need to compare to self
                continue
            if j in dominated: # don't bother comparing against something already dominated.
                continue
            if strictlyDominated(strat1, strat2):
                dominated.append(i)
                break
    return dominated

print(findDominatedStrats(prisoners_dilemma, 0)) # should print [1]

def strictlyDominated(strat1, strat2, excluded=[]):
    for idx, (play1, play2) in enumerate(zip(strat1, strat2)):
        if idx in excluded:
            continue
        if play1 >= play2:
            return False
    return True

strategy0 = (3, 4, 9)
strategy1 = (6, 7, 1)

isDom = strictlyDominated(strategy0, strategy1)
print("Is {0} strictly dominated by {1}, everything considered? {2}".format(strategy0, strategy1, isDom))

excl = [2]
isDom = strictlyDominated(strategy0, strategy1, excl)
print("Excluding {0}, is {1} strictly dominated by {2}? {3}".format(excl, strategy0, strategy1, isDom))

def findDominatedStrats(game, player, selfExcluded=[], otherExcluded=[]):
    dominated = []
    strats = [strategyExtractor(game, player, s) for s in range(len(game))]
    for i, strat1 in enumerate(strats):
        for j, strat2 in enumerate(strats):
            if i == j or i in selfExcluded or j in selfExcluded: # no need to compare to self or already-known
                continue
            if j in dominated:
                continue
            if strictlyDominated(strat1, strat2, otherExcluded):
                dominated.append(i)
                break # once dominated once, we no longer care, so just move on to next strategy
    return dominated

print(findDominatedStrats(prisoners_dilemma, 0, otherExcluded=[1])) # should print [1]

def iterativeDominatedRemoval(game):
    player0Dominated = []
    player1Dominated = []
    while True:
        changes = False
        newlyDominated0 = findDominatedStrats(game, 0, player0Dominated, player1Dominated)
        if newlyDominated0:
            player0Dominated.extend(newlyDominated0)
            changes = True
        newlyDominated1 = findDominatedStrats(game, 1, player1Dominated, player0Dominated)
        if newlyDominated1:
            player1Dominated.extend(newlyDominated1)
            changes = True
        if changes == False:
            break
    nonDominatedStrats = [(x, y) for x in range(len(game)) for y in range(len(game))
                         if x not in player0Dominated if y not in player1Dominated]
    return nonDominatedStrats

print("PD: nondominated strategy pairs:", iterativeDominatedRemoval(prisoners_dilemma))
print("S0: nondominated strategy pairs:", iterativeDominatedRemoval(solvable_0))
print("S1: nondominated strategy pairs:", iterativeDominatedRemoval(solvable_1))
print("S2: nondominated strategy pairs:", iterativeDominatedRemoval(solvable_2))

print("NS0: nondominated strategy pairs:", iterativeDominatedRemoval(not_solvable_0))
print("NS1: nondominated strategy pairs:", iterativeDominatedRemoval(not_solvable_1))
print("NS2: nondominated strategy pairs:", iterativeDominatedRemoval(not_solvable_2))

# fill out this function with your final code.
# return (x, y) as the best STRATEGIES or None.
def solve_game( game, verbose=False):
    solutions = iterativeDominatedRemoval(game)
    if len(solutions) == 1:
        if verbose:
            print("Solution of Nash equilibrium exists: strategy {s[0]} for player 0 "
                  "and strategy {s[1]} for player 1".format(s=solutions[0]))
        return solutions[0]
    elif verbose:
        print("No nash equilibrium strategy, there are still {0} possible options.".format(len(solutions)))
    return None

gameDict = {"s0" : solvable_0, "s1" : solvable_1, "s2" : solvable_2, 
            "ns0" : not_solvable_0, "ns1" : not_solvable_1, "ns2" : not_solvable_2,
            "pd" : prisoners_dilemma}
verbosity = True
for game in ("pd", "s0", "s1", "s2", "ns0", "ns1", "ns2"):
    print("\nTrying to solve {0}".format(game))
    solutions = solve_game(gameDict[game], verbosity)


