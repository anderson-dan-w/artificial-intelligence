from __future__ import print_function
import itertools

class Game(object):
    """ Game-theoretical set of methods to attach to an already-existing
        matrix of strategies for players
    """
    def __init__(self, matrix):
        self.matrix = matrix
        self.nstrats = len(self.matrix)  ## matrix is a square 2D array

    def payoff(self, position0, position1):
        """ Determines the outcome if player 0 is in @aposition0 and
            player 1 is in @a position1
        """
        return self.matrix[position0][position1]

    @staticmethod
    def exists_domination(strategy0, strategy1):
        """ Determines if one strategy dominates another. Returns 0 if
            strategy0 dominates, 1 if strategy1 dominates, and -1 if neither
        """
        dominate0 = all(p0 > p1 for p0, p1 in zip(strategy0, strategy1))
        dominate1 = all(p1 > p0 for p0, p1 in zip(strategy0, strategy1))
        return 0 if dominate0 else 1 if dominate1 else -1

    def extract_strategy(self, player, strategy):
        """  Extracts the @a strategy for the given @a player from the @a matrix
        """
        if player not in (0, 1):
            raise Exception("player must be 0 or 1")
        ## this looks gross
        if player == 0:
            extracted = [self.matrix[strategy][i][player]
                         for i in range(self.nstrats)]
        else:
            extracted = [self.matrix[i][strategy][player]
                         for i in range(self.nstrats)]
        return extracted

    @staticmethod
    def is_strictly_dominated(strategy0, strategy1, known1):
        """ Determines if strategy0 is strictly dominated (always inferior,
            never even equal) to strategy1
        """
        return all(p0 < p1 for p0, p1 in zip(strategy0, strategy1))

    def discover_dominated(self, player, known0=None, known1=None):
        """ Determines which strategies are strictly dominated """
        if known0 is None:
            known0 = set()
        if known1 is None:
            known1 = set()
        dominated = set()
        strategies = [self.extract_strategy(player, s)
                      for s in range(self.nstrats)]
        for s0, s1 in itertools.combinations(strategies, 2):
            ## previously discovered as dominated
            if s0 in known0 or s1 in known0:
                continue
            ## don't compare to already-dominated
            if s1 in dominated:
                continue
            if not self.is_strictly_dominated(s0, s1, known1):
                continue
            dominated.add(s0)
        return dominated

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

