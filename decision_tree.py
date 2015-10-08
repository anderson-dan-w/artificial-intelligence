"""
decision_tree.py

Using decision trees to analyze data and come to (probabilistic?) answer to:
"given this setup, if presented with new item, containing attributes X, Y, Z,
is outcome variable W likely to be True or False"

"""
from __future__ import print_function, division
import collections
import math


class DecisionTree(object):
    """ Instantiate a Decision Tree that attempts to make optimal choices
        considering each attribute, weighing the outcome of any particular
        attribute being set to each label, and describing how that affects
        the certainty of the outcome.
    """
    def __init__(self, data, dependent_idx, dependent_success, attr_mapper):
        self.data = data
        self.dependent_idx = dependent_idx
        self.dependent_success = dependent_success
        self.attr_mapper = attr_mapper
        self.not_success = self._not_success()
        self.nattrs = len(self.data[0])  ## length of first row
        self.used_idxs = set([self.dependent_idx])
        self.decision, self.certainty = self.get_certainty(self.data)
        self.root = Node(self.data, self)

    def _not_success(self):
        """ Convenience method to establish some non-success decision output.
            While this does not require the data to have a binary dependent
            variable, it treats decisions as such - either it was or it was
            not the successful outcome.
        """
        not_success = -1
        if self.dependent_success == -1:
            not_success = 0
        outcome_labels = self.attr_mapper[self.dependent_idx]["labels"]
        if len(outcome_labels) == 2:
            not_success = [l for l in outcome_labels
                             if l != self.dependent_success][0]
        return not_success

    def get_groupby(self, data_slice, attr_idx):
        """ return generators containing subsets of data, split by labels of
            the given attr_idx
        """
        ## @TODO: multiple passes through data...
        for label in self.attr_mapper[attr_idx]["labels"]:
            yield (label, (row for row in data_slice if row[attr_idx] == label))

    def get_positives(self, data_slice):
        """ Create list of all rows of data slice that are considered to have
            the positive outcome for the dependent variable
        """
        return [row for row in data_slice
                if row[self.dependent_idx] == self.dependent_success]

    def calculate_entropy(self, data_slice, attr_idx):
        """ Determine the entropy of a given attribute in relation to how it
            predicts the dependent variable, using the standard "shannon"
            measure of entropy.
        """
        nitems = len(data_slice)
        weighted_entropy = 0
        ## grab each possible outcome for the given attr
        for _, subiter in self.get_groupby(data_slice, attr_idx):
            ## find out how it corresponds to variable of interest
            counts = collections.Counter(row[self.dependent_idx]
                                         for row in subiter)
            subtotal = sum(counts.values())
            ## entropy definition: (negative)(sum all i)p(i)*log(p(i))
            entropy = 0
            for value in counts.values():
                local_weight = value / subtotal
                entropy += local_weight * math.log(local_weight, 2)
            weighted_entropy += (subtotal / nitems) * -entropy
        return weighted_entropy

    def find_min_entropy(self, data_slice, used_idxs):
        """ Find the attribute that introduces the least amount of entropy in
            regards to the dependent variable. That is, the more homogeneous the
            dependent output for each label of an attribute, the lower the
            entropy which makes it more helpful to use for decision making.
        """
        ## since used_idxs changes on per-node basis, recalculate unused_idxs
        unused_idxs = set(range(self.nattrs)).difference(used_idxs)
        entropies = {attr_idx: self.calculate_entropy(data_slice, attr_idx)
                     for attr_idx in unused_idxs}
        ## return the one with least entropy
        return min(entropies.items(), key=lambda x: x[1])

    def get_certainty(self, data_slice):
        """ Calculate the "average" outcome for this data set, either success
            or failure, and report the certainty of that decision
        """
        nitems = len(data_slice)
        npositives = len(self.get_positives(data_slice))
        probability = npositives / nitems
        if probability >= 0.5:  ## half the time or more, this is "good"
            decision = self.dependent_success
            certainty = probability
        else:
            decision = self.not_success
            certainty = 1 - probability
        return decision, certainty

    def get_decision(self, row):
        """ Provided a row of data, with or without the dependent variable
            included, predict the outcome according to the already-instantiated
            @a DecisionTree and report the decision, the certainty of it,
            the number of items used to determine it, and the path taken to
            get there.
        """
        if len(row) != self.nattrs:
            row = row[:]
            ## placeholder; should be problem-specific what's a better default
            row.insert(self.dependent_idx, self.not_success)
        node, nitems = self.root, len(self.data)
        decision, certainty = self.decision, self.certainty
        path = []
        while True:
            label = row[node.attr]
            branch = node.branches[label]
            if branch.nitems == 0:
                break
            decision, certainty = branch.decision, branch.certainty
            nitems = branch.nitems
            path.append((node.attr, label))
            if branch.child is None:
                break
            node = branch.child
        return decision, certainty, nitems, path

    def describe_path(self, path):
        """ Print explanation of each node considered in order to get to
            a particular decision
        """
        output = "PATH:"
        for attr_idx, label in path:
            attr_dict = self.attr_mapper[attr_idx]
            print(attr_dict)
            output += "\n  {}={}".format(attr_dict["name"], label)
        return output


class Node(object):
    """ Node in @a DecisionTree, representing the most helpful attribute to
        consider with the remaining data.
        Contains labelled @a Branch-es to dig further into @a DecisionTree.
    """
    def __init__(self, data, decision_tree, parent_branch=None):
        ## NB: space-optimization would not have copies of (subsets of) data
        ## running around. Example - if it were a pandas.DataFrame, this would
        ## just be a 1-d array of bools saying which rows to consider (filter).
        self.data = data
        self.decision_tree = decision_tree
        self.parent_branch = parent_branch  ## ok to be None, ie root
        ## NB: set(set) = copy; shallow copy ok since only ints as idxs
        if self.parent_branch is not None:
            self.used_idxs = set(self.parent_branch.parent_node.used_idxs)
        else:
            self.used_idxs = set(self.decision_tree.used_idxs)

        self.attr, self.entropy = self.decision_tree.find_min_entropy(
            self.data, self.used_idxs)
        self.used_idxs.add(self.attr)

        self.branches = {}
        ## grab each possible outcome for the given attr
        for label, subiter in self.decision_tree.get_groupby(self.data,
                self.attr):
            subdata = list(subiter)  ## unroll generator
            self.branches[label] = Branch(label, subdata, self)

class Branch(object):
    """ Create an offshoot from a Node, representing a specific label for
        whatever attr the @a Node dealt with. E.g. if @a Node.attr was related
        to "color", @a Branch.label might be "green" or "blue" or "red".
    """
    def __init__(self, label, data, parent_node):
        self.label = label
        self.data = data
        self.parent_node = parent_node
        self.decision_tree = self.parent_node.decision_tree
        self.child = None
        self.nitems = len(data)
        if self.nitems == 0:
            self.decision, self.certainty = "unknown", 0.0
            return

        self.decision, self.certainty = self.decision_tree.get_certainty(
            self.data)
        ## if attrs remain and not 100% certain, keep searching
        if self.certainty == 1.0:
            return
        if len(self.parent_node.used_idxs) != self.decision_tree.nattrs:
            self.child = Node(self.data, self.decision_tree, self)

