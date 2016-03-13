#!/usr/bin/env python3

## python modules
import random


def hillclimb_basic(attempt, goal):
    original = attempt
    idx, nsteps = 0, 1
    while calculate_distance(attempt, goal) > 0:
        while attempt[idx] == goal[idx]:
            idx += 1
        nudge_up = DOMAIN
        

## Let's take a generic example, just to prove the point:
## Can we get an initial string closer to a desired string through
## hill-climbing and genetic algorithms?

## need a domain...
DOMAIN = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "  ## with the space!

## ...and a goal...
GOAL = "METHINKS IT IS LIKE A WEASEL"

## ...and some sort of distance metric between a 'solution' and the GOAL...
def calculate_distance(attempt, goal):
    distance = 0
    for a, g in zip(attempt, goal):
        letter_distance = abs(DOMAIN.index(a) - DOMAIN.index(g))
        ## sneaky: loop the alphabet: Z => ' ' => A
        if letter_distance >= 14:
            letter_distance = 27 - letter_distance
        distance += letter_distance
    return distance

def nearby_letter(initial, direction=0):
    if direction == 0:
        direction = random.choice([-1, 1])
    if direction > 0:
        new_letter = DOMAIN[(DOMAIN.index(initial) + 1) % len(DOMAIN)]
    else:
        new_letter = DOMAIN[(DOMAIN.index(initial) - 1) % len(DOMAIN)]
    return new_letter

## Let's have a starting string:
START = "A" * len(GOAL)

