#!/usr/bin/python3

## python modules
import collections

class Node(object):
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

    ##======================================================================##
    def add_neighbors(self, neighbors):
        if not isinstance(neighbors, collections.Iterable):
            neighbors = set([neighbors])
        if not isinstance(next(iter(neighbors)), Node):
            neighbors = set(Node(node) for node in neighbors)
        self.neighbors.update(neighbors)
        return

    @property
    def unvisited_neighbors(self):
        return [node for node in self.neighbors if not node.visited]

    ##======================================================================##
    def show_path(self, sort_fn=None):
        for node in self.path:
            cost = ""
            if sort_fn:
                cost = " costing " + str(sort_fn(node))
            print(str(node) + cost)

###############################################################################
class Graph(object):
    def __init__(self, nodes=None, start=None, finish=None, sort_fn=None):
        if nodes is None:
            nodes = set()
        elif not isinstance(next(iter(nodes)), Node):
            nodes = set(Node(node) for node in nodes)
        self.nodes = nodes
        self.start = start
        self.finish = finish
        if sort_fn is None:
            sort_fn = greedy_sort 
        self.sort_fn = sort_fn

    ##======================================================================##
    def add_node(self, node):
        self.nodes.add(node)

    def add_nodes(self,nodes):
        self.nodes.update(nodes)

    ##======================================================================##
    def set_start(self, node):
        if node not in self.nodes:
            self.nodes.add(node)
        self.start = node

    def set_finish(self, node):
        if node not in self.nodes:
            self.nodes.add(node)
        self.finish = node

    ##======================================================================##
    def search(self):
        frontier = [self.start]
        while frontier:
            current_node = frontier.pop(0)  ## pop first element
            if current_node == self.finish:
                current_node.path.append(current_node)
                return current_node
            current_node.visited = True
            neighbors = sorted(current_node.unvisited_neighbors,
                                key=self.sort_fn)
            for n in neighbors:
                if n in frontier:
                    continue    ## skip those already in queue
                n.path = current_node.path + [current_node]
                n.distance = current_node.distance + 1
                n.total_weight = current_node.total_weight + n.weight
                frontier.append(n)
            frontier.sort(key=self.sort_fn)
        return None


##############################################################################
## Sort functions:
def uniform_cost_sort(node):
    return node.total_weight

def greedy_sort(node):
    return node.weight

def a_star(node):
    return node.total_weight + node.heuristic

## An admissible heuristic might be, say, the manhattan distance, if the
## nodes were arranged on a Grid, because that would always undershoot, but
##  would hopefully come close-ish to estimating the remaining cost.

###############################################################################
def test():
    ns, nf = Node("Start"), Node("Finish")
    n1, n2, n3, n4 = Node("A"), Node("B"), Node("C"), Node("D")
    ns.add_neighbors([n1, n2])
    n2.add_neighbors(n3)
    n3.add_neighbors(n4)
    n4.add_neighbors(nf)
    n1.add_neighbors(nf)

    graph = Graph([ns, n1, n2, n3, n4, nf], start=ns, finish=nf)
    return graph.search()

def test_greedy():
    ns, nf = Node("Start"), Node("Finish")
    n1, n2, n3, n4 = Node("A", 100), Node("B"), Node("C"), Node("D")
    ns.add_neighbors([n1, n2])
    n2.add_neighbors(n3)
    n3.add_neighbors(n4)
    n4.add_neighbors(nf)
    n1.add_neighbors(nf)

    graph = Graph([ns, n1, n2, n3, n4, nf], start=ns, finish=nf)
    return graph.search()

def test_uniform():
    ns, nf = Node("Start"), Node("Finish")
    n1, n2, n3, n4 = Node("A", 10), Node("B", 8), Node("C", 8), Node("D", 8)
    ns.add_neighbors([n1, n2])
    n2.add_neighbors(n3)
    n3.add_neighbors(n4)
    n4.add_neighbors(nf)
    n1.add_neighbors(nf)

    graph = Graph([ns, n1, n2, n3, n4, nf], start=ns, finish=nf,
            sort_fn=uniform_cost_sort)
    return graph.search()

##############################################################################
ns, nf = Node("Start"), Node("Finish")
n1, n2, n3, n4 = Node("A"), Node("B"), Node("C"), Node("D")
ns.add_neighbors([n1, n2])
n2.add_neighbors(n3)
n3.add_neighbors(n4)
n4.add_neighbors(nf)
n1.add_neighbors(nf)
nodes = [ns, n1, n2, n3, n4, nf]
