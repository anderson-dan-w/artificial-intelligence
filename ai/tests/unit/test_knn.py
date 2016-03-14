import unittest

from ai.k_nearest_neighbors import *

class TestKNN(unittest.TestCase):
    def setUp(self):
        ## ignore for now; every row starts with some output value
        self.SOME_OUTPUT = 0

    def test_calc_distance_one_dimension(self):
        distance = 3
        dim1_vector = [self.SOME_OUTPUT, distance]
        dim1_origin = [self.SOME_OUTPUT, 0]
        self.assertAlmostEqual(distance,
                               calc_distance(dim1_vector, dim1_origin))
        ## for good measure, test commutativity
        self.assertAlmostEqual(distance,
                               calc_distance(dim1_origin, dim1_vector))

    def test_calc_distance_multi_dimension(self):
        dim2_vector = [self.SOME_OUTPUT, 3, 4]
        dim2_origin = [self.SOME_OUTPUT, 0, 0]
        distance = math.sqrt(sum(v**2 for v in dim2_vector))
        self.assertAlmostEqual(distance,
                               calc_distance(dim2_vector, dim2_origin))
        ## for good measure, test commutativity
        self.assertAlmostEqual(distance,
                               calc_distance(dim2_origin, dim2_vector))


