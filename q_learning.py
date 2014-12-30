##pr6
atlas=amap ##(reference to pr1!!)
import random
import copy

gamma = 0.9
n = len( atlas)

states = [(x,y) for x in range(0, n) for y in range(0, n)]
actions = ["^", "v", ">", "<"]
possibilities = {
  "^": [(0,-1), (1,0), (0,1), (-1,0)],
  "v": [(0,1), (0,-1), (1,0), (-1,0)],
  ">":  [(1,0), (0,1), (0,-1), (-1,0)],
  "<":  [(-1,0), (1,0), (0,1), (0,-1)]
}

probabilities = [0.7, 0.1, 0.1, 0.1]

def successors( s, a):
    n = len( atlas) - 1
    result = []
    x, y = s
    for dx, dy in possibilities[ a]:
        new_x = x + dx
        new_y = y + dy
        if new_x < 0 or new_x > n or new_y < 0 or new_y > n or atlas[ new_y][ new_x] == 'x':
            result.append( s) # couldn't move, so stay at current state
        else:
            result.append((new_x, new_y))
    return result

print(successors((0, 0), "^"))
print(successors((5, 10), "v"))

goal = (26, 26)

def r(s, a):
    if s == goal:
        return 10000
    first_successor = successors( s, a)[ 0]
    if s == first_successor:
        return -100
    x, y = first_successor
    terrain = atlas[ y][ x]
    return -costs[ terrain]

print(r((0, 0), "v"))
print(r((0, 0), "^"))
print(r((5, 10), ">"))
print(r((26, 26), ">"))

def best_action(q, s):
    a = actions[ 0]
    value = q[ s][ a]
    for action in actions[1:]:
        if q[ s][ action] > value:
            value = q[ s][ action]
            a = action
    return a

def error( v1, v2):
    largest_diff = 0.0
    for state in states:
        e = abs( v1[ state] - v2[ state])
        if e > largest_diff:
            largest_diff = e
    return largest_diff

q = {}
for state in states:
    q[ state] = {}
    for action in actions:
        q[state][ action] = 0.0

def one_value_iteration( last):
    policy = {}
    current = copy.deepcopy( last)
    for state in states:
        x, y = state
        if atlas[ y][ x] == 'x':
            continue
        for action in actions:
            ss = successors( state, action)
            vs = map( lambda s: last[ s], ss)
            expected_v_primes = map( lambda x: x[ 0] * x[ 1], zip( probabilities, vs))
            q[ state][ action] = r( state, action) + gamma * sum( expected_v_primes)
        ba = best_action( q, state)
        policy[ state] = ba
        current[ state] = q[ state][ ba]
    return (current, policy, q)

def value_iteration():
    last = {}
    for state in states:
        last[ state] = 0.0
    
    while True:
        current, policy, q = one_value_iteration( last)
        e = error( current, last)
        last = current
        if e < 0.000001:
            return policy

if False:  ## toggle between True or False to do or not-do ValueIteration
    policy = value_iteration()
    for i in range( 0, n):
        for j in range( 0, n):
            if (j, i) in policy:
                print(policy[(j, i)], end=" ")
            else:
                print("x", end=" ")
        print()

action_lookup = {(0, -1): "^", (0, 1): "v", (1, 0): ">", (-1, 0): "<"}

def get_moves(state):
    n = len(atlas) - 1
    moves = []
    (x, y) = state
    for (dx, dy), action in action_lookup.items():
        new_x, new_y = (x + dx, y + dy)
        if (new_x < 0 or new_x > n or new_y < 0 or new_y > n or
            atlas[new_y][new_x] == 'x'):
            moves.append((action, state))
        else:
            moves.append((action, (new_x, new_y)))
    return moves

print(get_moves((0, 0)))
print(get_moves((22, 3)))

def get_q_sorted_moves(q, state, moves=None):
    if moves is None:
        moves = get_moves(state)
    return sorted(moves, key=lambda m: q[state][m[0]], reverse=True)

def get_q_move(q, state):
    moves = get_moves(state)
    if random.random() < epsilon:
        return random.choice(moves)
    sorted_moves = get_q_sorted_moves(q, state, moves)
    move = sorted_moves[0]
    #rand = random.random()
    #if rand > .9:
    #    move = sorted_moves[3]
    #elif rand > .8:
    #    move = sorted_moves[2]
    #elif rand > .7:
    #    move = sorted_moves[1]
    return move

def calculate_reward(state, move):
    if state == goal:
        return 10000.0
    (action, (new_x, new_y)) = move
    if state == (new_x, new_y):  ## oh no, we didn't move!
        return -100.0
    terrain = atlas[new_y][new_x]
    return -1 * costs[terrain]

alpha = 0.25
epsilon = 0.1  ## this was arbitrary
gamma = 0.98

def do_q_learning(num_episodes, DEBUG=False):
    ## dictionary comprehensions are nice
    q = {state: {action: 0.0 for action in actions} for state in states}
    for action in actions:
        q[goal][action] = 100000.0
    policy = {state: 'x' for state in states} # set the default policy
    for i in range(num_episodes):
        state = (0, 0) ## being lazy, starting at origin
        while state != goal:
            move = get_q_move(q, state)
            reward = calculate_reward(state, move)
            action, s_prime = move
            q_prime_best = max(q[s_prime][a] for (a, s_pp) in get_moves(s_prime))
            q[state][action] = ((1-alpha) * q[state][action] +
                                alpha * (reward + gamma * q_prime_best))
            # the return from sorted_moves has the best move in [0], and the action
            # is stored in move[0]; so the best action is in moves[0][0]
            policy[state] = get_q_sorted_moves(q, state)[0][0]
            state = s_prime
        if DEBUG: print("Finished episode {}".format(i))
    return q, policy

def print_policy(policy, q=None):
    if q is not None:
        for state in sorted(states, key=lambda x: (x[1], x[0])):
            print("({:02}, {:02})".format(state[0], state[1]), end="\t")
            q_dict = {action: q[state][action] for action in actions}
            for action, value in sorted(q_dict.items(), key=lambda x:x[1], reverse=True):
                print("{a} {v:.6}".format(a=action, v=value), end="\t")
            print() ##to get a new line
    for i in range(len(atlas)):
        for j in range(len(atlas)):
            print("{p}".format(p=policy[(j, i)]), end="")
        print() ## to get a new line
    return  

num_episodes = 250
q, policy = do_q_learning(num_episodes, DEBUG=True)
print_policy(policy, q)



