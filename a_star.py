#!/usr/bin/python3
""" a_star.py
    using best-first search algorithm with heuristics to improve upon greedy
"""
from __future__ import print_function, division
import collections

class Node(object):
    """ Element of the @a Graph to be visited/traversed/evaluated.
    """
    def __init__(self, info, weight=1, heuristic=0):
        self.info = info
        self.weight = weight
        self.total_weight = 0
        self.heuristic = heuristic
        self.visited = False
        self.distance = 0
        self.path = []
        self.neighbors = set()

    def __str__(self):
        return str(self.info)

    def add_neighbors(self, neighbors):
        """ Add/update neighbor list for @a Node, being smart enough to convert
            neighbors into a set of @a Nodes if necessary.
        """
        if not isinstance(neighbors, collections.Iterable):
            neighbors = set([neighbors])
        if not isinstance(next(iter(neighbors)), Node):
            neighbors = set(Node(info) for info in neighbors)
        self.neighbors.update(neighbors)
        return

    @property
    def unvisited_neighbors(self):
        """ Generate list of neighbor @a Nodes who haven't been visited """
        return [node for node in self.neighbors if not node.visited]

    def show_path(self, sort_fn=None):
        """ return str with each node in order after running search """
        output = ""
        for node in self.path:
            cost = " costing {}".format(sort_fn(node)) if sort_fn else ""
            output += "{} {}\n".format(node, cost)
        return output


###################################
class Graph(object):
    """ @a Node container holding various information, including @a start,
        @a finish and @a sort_fn, which dictates the heuristic to be used
    """
    def __init__(self, nodes=None, start=None, finish=None, sort_fn=None):
        if nodes is None:
            nodes = set()
        elif not isinstance(nodes, collections.Iterable):
            nodes = set([nodes])
        if nodes and not isinstance(next(iter(nodes)), Node):
            nodes = set(Node(node) for node in nodes)
        self.nodes = nodes
        self.start = start
        self.finish = finish
        if sort_fn is None:
            sort_fn = greedy_sort
        self.sort_fn = sort_fn

    def add_node(self, node):
        """ add @a Node to @a Graph. Each @a Node contains info about its
            neighbors, so @a Graph isn't responsible for any bookkeeping there
        """
        self.nodes.add(node)

    def add_nodes(self, nodes):
        """ convenience function to add multiple @a Nodes at once """
        if not isinstance(nodes, collections.Iterable):
            nodes = set([nodes])
        self.nodes.update(nodes)

    def set_start(self, node):
        """ @a Graphs are traversed by moving from start @a Node to finish
            @a Node; this enables the @a Graph to know where to start.
            Also allows for @a Node that was not yet in the @a Graph
        """
        if node not in self.nodes:
            self.nodes.add(node)
        self.start = node

    def set_finish(self, node):
        """ @a Graphs are traversed by moving from start @a Node to finish
            @a Node; this enables the @a Graph to know where to finish.
            Also allows for @a Node that was not yet in the @a Graph
        """
        if node not in self.nodes:
            self.nodes.add(node)
        self.finish = node

    def search(self):
        """ Run search algorithm, on @a Graph, traversing from @a start
            to @a finish, which must be pre-defined.
        """
        frontier = [self.start]
        while frontier:
            current_node = frontier.pop(0)  ## pop first element
            if current_node == self.finish:
                current_node.path.append(current_node)
                return current_node
            current_node.visited = True
            neighbors = sorted(current_node.unvisited_neighbors,
                                key=self.sort_fn)
            for neighbor in neighbors:
                if neighbor in frontier:
                    continue    ## skip those already in queue
                neighbor.path = current_node.path + [current_node]
                neighbor.distance = current_node.distance + 1
                neighbor.total_weight = (current_node.total_weight +
                                         neighbor.weight)
                frontier.append(neighbor)
            frontier.sort(key=self.sort_fn)


##############################################################################
## Sort functions:
def uniform_cost_sort(node):
    """ most naive search algorithm - consider the entire cost/weight of the
        path when choosing next @a Node
    """
    return node.total_weight

def greedy_sort(node):
    """ search algorithm - consider only the cost/weight of the next @a Node
        when choosing
    """
    return node.weight

## An admissible heuristic might be, say, the manhattan distance, if the
## nodes were arranged on a Grid, because that would always undershoot, but
##  would hopefully come close-ish to estimating the remaining cost.
def a_star(node):
    """ search algorithm - consider both the entire cost/weight of the path,
        and some heuristic value to estimate the remaining cost when choosing
        next @a Node
    """
    return node.total_weight + node.heuristic
