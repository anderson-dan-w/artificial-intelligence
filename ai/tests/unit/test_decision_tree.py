from __future__ import print_function
import sys
import unittest

try:
    from ai import decision_tree
except:
    raise Exception("syspath:\n\t{}".format("\n\t".join(sys.path)))

if sys.version_info.major == 2:
    from StringIO import StringIO
else:
    from io import StringIO

def get_stream_and_reporter():
    stream = StringIO()
    return stream, stream.write

class TestDecisionTree(unittest.TestCase):
    def setUp(self):
        self.data = [[0, 1, 1, 1],
                     [0, 1, 2, 2],
                     [1, 2, 2, 2],
                     [1, 2, 2, 1]]
        self.attr_map = [
            {"name": "dep", "labels": [0, 1]},
            {"name": "one", "labels": [1, 2]},
            {"name": "two", "labels": [1, 2]},
            {"name": "three", "labels": [1, 2, 3]}
        ]
        self.decision_tree = decision_tree.DecisionTree(
            self.data, 0, 1, self.attr_map)

    def test_calculate_entropy(self):
        ## attr 1 should be perfect - it describes dependent 100%
        entropy1 = self.decision_tree.calculate_entropy(self.data, 1)
        self.assertAlmostEqual(0.0, entropy1)
        ## attr 3 should be bad - it is 50-50 in either dependent case
        entropy3 = self.decision_tree.calculate_entropy(self.data, 3)
        self.assertAlmostEqual(1.0, entropy3)
        entropy2 = self.decision_tree.calculate_entropy(self.data, 2)
        self.assertGreater(entropy2, entropy1)
        self.assertLess(entropy2, entropy3)

    def test_find_min_entropy(self):
        used_idxs = [self.decision_tree.dependent_idx]
        attr, entropy = self.decision_tree.find_min_entropy(
            self.data, used_idxs)
        self.assertEqual(1, attr)
        self.assertAlmostEqual(0.0, entropy)

    def test__not_success(self):
        ## @a DecisionTree._not_success() called on __init__()
        self.assertEqual(0, self.decision_tree.not_success)

        ## if >2 options, defaults to some generic value, -1
        self.decision_tree.attr_mapper[
            self.decision_tree.dependent_idx]["labels"].append(3)
        self.assertEqual(-1, self.decision_tree._not_success())
        ## but changes to another generic value (0) if -1 is dependent_success
        self.decision_tree.dependent_success = -1
        self.assertEqual(0, self.decision_tree._not_success())

    def test_get_groupby(self):
        expected_counts = {1: 2, 2:2, 3:0}
        counts = {}
        for label, subslice in self.decision_tree.get_groupby(self.data, 3):
            counts[label] = len(list(subslice))
        self.assertEqual(expected_counts, counts)

    def test_get_positives(self):
        all_positive = self.decision_tree.get_positives(self.data)
        self.assertEqual(2, len(all_positive))

        attr2eq1_slice = [r for r in self.data if r[2] == 1]
        attr2eq1_positive = self.decision_tree.get_positives(attr2eq1_slice)
        self.assertEqual(0, len(attr2eq1_positive))

    def test_get_certainty(self):
        decision, certainty = self.decision_tree.get_certainty(self.data)
        self.assertEqual((self.decision_tree.dependent_success, 0.5),
                         (decision, certainty))

        attr2eq1_slice = [r for r in self.data if r[2] == 1]
        decision, certainty = self.decision_tree.get_certainty(attr2eq1_slice)
        self.assertEqual((self.decision_tree.not_success, 1.0),
                         (decision, certainty))

    def test_get_decision(self):
        row = [0, 2, 1, 1]
        d, c, n, p = self.decision_tree.get_decision(row)
        ## @TODO tbh, not sure why nitems==2
        self.assertEqual((1, 1.0, 2), (d, c, n), p)

        ## if test row comes without dependent variable
        xrow = [2, 1, 1]
        xd, xc, xn, xp = self.decision_tree.get_decision(row)
        self.assertEqual((xd, xc, xn, xp), (d, c, n, p))

    def test_describe_path(self):
        d, c, n, p = self.decision_tree.get_decision([0, 2, 1, 1])
        self.assertRegex(self.decision_tree.describe_path(p), "PATH:\n[(1, 2)]")

class TestNode(unittest.TestCase):
    pass

class TestBranch(unittest.TestCase):
    pass
