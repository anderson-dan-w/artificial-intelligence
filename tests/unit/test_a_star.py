""" unit testing for a_star search (and Node and Graph basics) """
from __future__ import print_function, absolute_import
import unittest

try:
    from ai import a_star
except:
    import sys
    raise Exception("Syspath=\n\t{}".format("\n\t".join(sys.path)))

class TestNode(unittest.TestCase):
    """ test various @a Node aspects """
    def setUp(self):
        self.node = a_star.Node({"name": "base node"})
        self.new_node = a_star.Node({"name": "new node"})

    def test_add_neighbors(self):
        self.assertEqual(set(), self.node.neighbors)
        ## add one node
        self.node.add_neighbors(self.new_node)
        self.assertEqual(set([self.new_node]), self.node.neighbors)
        ## add non-node
        self.node.add_neighbors({"name": "second new node"})
        self.assertEqual(2, len(self.node.neighbors))
        ## add multiple nodes at once
        self.node.add_neighbors([{"name": "third"}, {"name": "fourth"}])
        self.assertEqual(4, len(self.node.neighbors))
        self.assertTrue(all(isinstance(node, a_star.Node)
                            for node in self.node.neighbors))

    def test_unvisited_neighbors(self):
        self.node.add_neighbors(self.new_node)
        self.assertEqual([self.new_node], self.node.unvisited_neighbors)
        self.new_node.visited = True
        self.assertEqual([], self.node.unvisited_neighbors)

    def test_show_path(self):
        self.assertEqual("", self.node.show_path())
        self.node.path.append(self.new_node)
        self.assertRegex(self.node.show_path(), "new node")
        dummy_sort = lambda x: 42
        self.assertRegex(self.node.show_path(dummy_sort), "new node.*42")


class TestGraph(unittest.TestCase):
    def setUp(self):
        self.node1 = a_star.Node({"name": 1})
        self.node2 = a_star.Node({"name": 2})
        self.node3 = a_star.Node({"name": 3})
        self.graph = a_star.Graph()

    def test_init(self):
        ## accepts single node
        graph1 = a_star.Graph(self.node1)
        self.assertEqual(set([self.node1]), graph1.nodes)
        ## accepts iterable of nodes
        graph2 = a_star.Graph(set([self.node1]))
        self.assertEqual(set([self.node1]), graph2.nodes)
        ## accepts non-nodes
        graph3 = a_star.Graph(self.node1.info)
        self.assertEqual(1, len(graph3.nodes))
        self.assertIsInstance(next(iter(graph3.nodes)), a_star.Node)

    def test_add_node(self):
        self.assertEqual(set(), self.graph.nodes)
        self.graph.add_node(self.node1)
        self.assertEqual(set([self.node1]), self.graph.nodes)

    def test_add_nodes(self):
        self.assertEqual(set(), self.graph.nodes)
        self.graph.add_nodes([self.node1, self.node2])
        self.assertEqual(set([self.node1, self.node2]), self.graph.nodes)
        ## test @a add_nodes is OK with getting non-iterable, ie one node
        self.graph.add_nodes(self.node3)
        self.assertEqual(set([self.node1, self.node2, self.node3]),
                         self.graph.nodes)

    def test_set_start(self):
        ## works if @a Node not in @a Graph already
        self.assertIsNone(self.graph.start)
        self.graph.set_start(self.node1)
        self.assertEqual(self.node1, self.graph.start)
        self.assertIn(self.node1, self.graph.nodes)

        ## works if @a Node already in @a Graph
        self.graph.add_node(self.node2)
        self.graph.set_start(self.node2)
        self.assertEqual(self.node2, self.graph.start)
        ## show it didn't double-add @a self.node2
        self.assertEqual(2, len(self.graph.nodes))

    def test_set_finish(self):
        ## works if @a Node not in @a Graph already
        self.assertIsNone(self.graph.finish)
        self.graph.set_finish(self.node1)
        self.assertEqual(self.node1, self.graph.finish)
        self.assertIn(self.node1, self.graph.nodes)

        ## works if @a Node already in @a Graph
        self.graph.add_node(self.node2)
        self.graph.set_finish(self.node2)
        self.assertEqual(self.node2, self.graph.finish)
        ## show it didn't double-add @a self.node2
        self.assertEqual(2, len(self.graph.nodes))

    def test_search__basic(self):
        self.graph.add_nodes([self.node1, self.node2, self.node3])
        self.node1.add_neighbors([self.node2, self.node3])
        self.node2.add_neighbors(self.node3)
        self.graph.set_start(self.node1)
        self.graph.set_finish(self.node3)

        ## show that everything is clean, ie unvisited
        self.assertTrue(not any(node.visited) for node in self.graph.nodes)
        ## run some search, whatever it is
        self.graph.search()
        ## prove that nodes have been visited
        self.assertTrue(all(node.visited) for node in self.graph.nodes)

    def search_helper(self, sort_fn, weights, heuristics=None):
        ## we need a little more complexity here: a graph such as:
        ##   2--3---
        ##  /       \
        ## 1--4--5--6
        ## should have differing search results based on the costs
        ## of nodes 2, 3, 4, 5 and the search function algorithm.
        node4 = a_star.Node({"name": 4})
        node5 = a_star.Node({"name": 5})
        node6 = a_star.Node({"name": 6})
        nodes = [self.node1, self.node2, self.node3, node4, node5, node6]
        self.graph.add_nodes(nodes)
        self.node1.add_neighbors([self.node2, node4])
        self.node2.add_neighbors(self.node3)
        self.node3.add_neighbors(node6)
        node4.add_neighbors(node5)
        node5.add_neighbors(node6)
        for idx, node in enumerate(nodes):
            node.weight = weights[idx]
            if heuristics is not None:
                node.heuristic = heuristics[idx]
        self.graph.set_start(self.node1)
        self.graph.set_finish(node6)
        ## these are set up so that 1->2->3 should be winning path
        expected_weight = sum(node.weight for node in
                              [self.node1, self.node2, self.node3])
        self.graph.sort_fn = sort_fn
        ## actually perform test
        self.graph.search()
        self.assertEqual(expected_weight, self.graph.finish.total_weight,
                         self.graph.finish.show_path(self.graph.sort_fn))

    def test_uniform_cost(self):
        ## node4 "weighs more", so path should be 1->2->3->6
        ## since it takes the overall total weight at any given choice
        weights = [1, 1, 1, 4, 1, 1]
        self.search_helper(a_star.uniform_cost_sort, weights)

    def test_greedy_cost(self):
        ## greedy will take the 3s first, rather than the 4-1
        weights = [1, 3, 3, 4, 1, 1]
        self.search_helper(a_star.greedy_sort, weights)

    def test_a_star_cost(self):
        ## now the heuristic should out-weigh the total_weight
        weights = [1, 1, 1, 4, 1, 1]
        heuristics = [0, 2, 2, 0, 0, 0]
        self.search_helper(a_star.a_star, weights, heuristics)

if __name__ == "__main__":
    unittest.main()

