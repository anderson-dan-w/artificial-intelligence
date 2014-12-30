#!/usr/bin/python3
import math
import random

## made it a square matrix so I didnt have to be clever about indexing
distances = [[0, -10, -20, -5, -18],
             [-10, 0, -15, -32, -10],
             [-20, -15, 0, -25, -16],
             [-5, -32, -25, 0, -35],
             [-18, -10, -16, -35, 0]]
ncities = len(distances)

max_dist = 35   ## eyeballed it
bias = 75       ## should be larger than 2*max_dist
penalty = -100  ## should be larger than bias

## created a randomized grid
grid = [[random.randrange(2) for i in range(ncities)] for i in range(ncities)]
## grid[a][b] means city A at time B.

def calc_delta_consensus(i, j, grid=grid):
    consensus = bias
    ## if anything else in the column is already on...
    consensus += penalty * any([grid[x][j] for x in range(ncities) if x != i])
    ## if anything else in the row is already on...
    consensus += penalty * any([grid[i][x] for x in range(ncities) if x != j])
    ## calculate average distance from previous time-states?
    prev_col = [grid[x][(j-1) % ncities] for x in range(ncities)]
    if sum(prev_col): ## if there's anything on in previous time state
        consensus += sum(distances[idx][i] for idx, city_on 
                          in enumerate(prev_col) if city_on)/sum(prev_col)
    ## calculate average distance to subsequent time-states?
    next_col = [grid[x][(j+1) % ncities] for x in range(ncities)]
    if sum(next_col): ## if there's anything on in subsequent time state
        consensus += sum(distances[idx][i] for idx, city_on 
                          in enumerate(next_col) if city_on)/sum(next_col)
    return consensus

def should_turn_on(consensus, temp):
    ## want to turn on every time consensus is positive (think of it like
    ## simulated annealing, but uphill - anything positive, we take)
    ## OR randomly with probability of e^(cons/temp)
    if consensus > 0 or (random.random() < math.e**(consensus/temp)):
        return True
    return False

def update_grid(temp=75, grid=grid):
    ## pick a random city-time, see if it should turn on (stochastically)
    city, time = random.randrange(5), random.randrange(5)
    consensus = calc_delta_consensus(city, time)
    grid[city][time] = 1 if should_turn_on(consensus, temp) else 0
    return

def is_valid(grid=grid):
    ## quick check to see if we have an appropriate grid - exactly one 1 in
    ## every row, and exactly one 1 in every column
    ncities = len(grid)
    for row in grid:
        if sum(row) != 1:
            return False
    for i in range(ncities):
        if sum(grid[x][i] for x in range(ncities)) != 1:
            return False
    return True

def calc_distance(grid=grid):
    ncities = len(grid)
    if not is_valid(grid):
        ## return high number so we know it sucks
        return max_dist * ncities * 10
    cities = []
    ## dumb code to find which cities are selected, in order
    for i in range(ncities):
        col = [grid[x][i] for x in range(ncities)]
        cities.append(col.index(1))
    distance = 0
    ## get distance from city to next_city, remembering to wrap around to
    ## the first city at the end
    for idx in range(len(cities)):
        current_city = cities[idx]
        next_city = cities[(idx+1) % ncities]
        distance += distances[current_city][next_city]
    return -1 * distance

## could use some work - sometimes succeeds, sometimes fails. 
## not sure how to set the temp or cooling schedule very well
def solve(temp=75, iters=10, grid=grid):
    epoch = ncities ** 2
    for i in range(iters):
        print("Starting epoch w dist={}; temp={}".format(calc_distance(grid),
            temp))
        for i in range(epoch):
            update_grid(temp, grid)
        temp *= .90
    return
