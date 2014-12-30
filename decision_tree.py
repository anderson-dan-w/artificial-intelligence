##pr9
from __future__ import print_function
from __future__ import division
import math
import collections

def read_in_definitions(fname):
    with open(fname) as fh:
        lines = fh.read().split("\n")
    attributes = {"safe-to-eat?" : {"e" : "edible", "p" : "poisonous"}}
    attr_names = ["safe-to-eat?"]
    for line in lines:
        numIdx, nameIdx = line.index("."), line.index(":")
        name = line[numIdx+2:nameIdx]
        values = line[nameIdx+2:].replace(" ","").split(",")
        values = {v[v.index("=")+1:] : v[:v.index("=")] for v in values}
        if "?" not in values:
            values["?"] = "missing"
        attributes[name] = values
        attr_names.append(name)
    return attributes, attr_names

fname = "pr9_mushroom_attributes.txt"
attrs, attr_names = read_in_definitions(fname)
for idx, attr in enumerate(attr_names): print(idx, attr, attrs[attr].items(), end="\n\n")

def read_in_data(fname):
    with open(fname) as fh:
        lines = fh.read().split("\n")
    data = [l.split(",") for l in lines]
    return data

fname = "pr9_mushroom_data.txt"
data_set = read_in_data(fname)
print(data_set[0])

def create_descriptive(data):
    s = ""
    for idx, attr in enumerate(attr_names):
        s += attr + ": " + attrs[attr][data[idx]] + "\n"
    return s

print(create_descriptive(data_set[0]))

def find_min_entropy(data, used_attrs=[], debug=False):
    if "safe-to-eat?" not in used_attrs:
        used_attrs.append("safe-to-eat?")
    best_gain, best_attr = 1.1, None
    total = len(data)
    for idx, attr in enumerate(attr_names):
        if attr in used_attrs:
            continue
        weighted_entropy = 0
        for label in attrs[attr].keys():
            counts = collections.Counter(d[0] for d in data if d[idx] == label)
            subtotal = sum(counts.values())
            entropy = sum(-1*(v/subtotal) * math.log(v/subtotal, 2) for v in counts.values())
            weighted_entropy += (subtotal/total) * entropy
        if debug:
            print("{n}: {e}".format(n=attr, e=weighted_entropy))
        if weighted_entropy < best_gain:
            if debug:
                print("\tnew best attr:", attr)
            best_gain = weighted_entropy
            best_attr = attr
    return best_gain, best_attr

find_min_entropy(data_set, debug=True)

class Node(object):
    def __init__(self, dataset, used_attrs=[], parent_branch=None):
        self.data = dataset
        self.parent_branch = parent_branch
        self.used_attrs = used_attrs[:]     # make it a copy, always a copy!
        if "safe-to-eat?" not in self.used_attrs:
            self.used_attrs.append("safe-to-eat?")
        self.entropy, self.attr = find_min_entropy(self.data, self.used_attrs)
        self.attr_idx = attr_names.index(self.attr)
        self.used_attrs.append(attr)
        self.branches = {}
        for label in attrs[self.attr].keys():
            subdata = [d for d in self.data if d[self.attr_idx] == label]
            self.branches[label] = Branch(label, subdata, self.used_attrs, self)

class Branch(object):
    def __init__(self, label, label_data, used_attrs, parent_node):
        #print("making a branch for label {l}... from {p}".format(l=label, p=parent_node.attr))
        self.label = label
        self.used_attrs = used_attrs   # not modifying, so no need to copy
        self.parent_node = parent_node # had to come from somewhere...
        self.child_node = None
        self.data = label_data
        self.ndata = len(self.data)
        if self.ndata == 0:
            self.decision, self.certainty = "unknown", 0.0
            return
        self.safeness = sum(1 for d in self.data if d[0] == "e")/self.ndata
        if self.safeness >= 0.5:
            self.decision, self.certainty = "edible", self.safeness
        else:
            self.decision, self.certainty = "poisonous", 1 - self.safeness
        # we've got hetergeneous data, AND more attrs to investigate...
        if self.certainty != 1.0 and len(used_attrs) != len(attr_names):
            self.child_node = Node(self.data, self.used_attrs, self)

class DecisionTree(object):
    def __init__(self, data):
        self.data = data
        self.root_node = Node(data)
        self.safeness = sum(1 for d in self.data if d[0] == "e")/len(data)
        if self.safeness >= 0.5:
            self.decision, self.certainty = "edible", self.safeness
        else:
            self.decision, self.certainty = "poisonous", 1 - self.safeness
        
    def create_tree(self, node=None, indent=0):
        if node is None:
            node = self.root_node
        tree = ""
        if indent == 0:
            header = ""
        elif indent == 1:
            header = "|---> "
        else:
            header = "|     " * (indent-1) + "|---> "
        label_dict = node.branches
        width = max(len(v) for v in attrs[node.attr].values()) + 1
        for label, branch in sorted(label_dict.items(), key= lambda x:attrs[node.attr][x[0]]):
            tree += ("{h}{n}: {l:<{w}} ({b.decision}, {b.certainty}, {b.ndata})\n"
                    .format(h=header, n=node.attr.capitalize(), l=attrs[node.attr][label], 
                            w=width, b=branch))
            if branch.child_node is not None:
                tree += self.create_tree(branch.child_node, indent+1)
        return tree
    
    def get_decision(self, mushroom_info):
        if len(mushroom_info) == 22: # attrs, without "poisonous/edible"
            mushroom_info.insert(0, "unknown") # to keep indexing accurate.
        curr_node = self.root_node
        prev_decision, prev_certainty, prev_ndata = self.decision, self.certainty, len(self.data)
        path = []
        while True:
            curr_label = mushroom_info[curr_node.attr_idx]
            curr_branch = curr_node.branches[curr_label]
            # check for ndata == 0...
            if curr_branch.ndata == 0:
                return (prev_decision, prev_certainty, path, prev_ndata)
            path.append((curr_node.attr, curr_label))
            if curr_branch.child_node is None:
                return (curr_branch.decision, curr_branch.certainty, path, curr_branch.ndata)
            # No leaf-nodes/dead ends, so keep traversing the tree...
            prev_decision, prev_certainty = curr_branch.decision, curr_branch.certainty
            prev_ndata = curr_branch.ndata
            curr_node = curr_branch.child_node
        return # won't get here, but just to close scope more explicitly... perhaps re-work?
    
    def describe_path(self, path):
        if path == []:
            return "Since, nothing pertinent is known,"
        return "If " + " AND\n\t".join("{a} is {l}".format(a=attr, l=attrs[attr][label]) 
                                        for attr, label in path)
    
    def describe_outcome(self, outcome):
        decision, certainty, path, ndata = outcome
        s = self.describe_path(path)
        s += ("\nTHEN, with {c:.05} certainty, it is {d}, according to {n} examples"
              .format(c=certainty, d=decision, n=ndata))
        return s

shuffled_data = data_set[:] # first copy it...
random.shuffle(shuffled_data)
ndata = len(shuffled_data)
ntraining = int(ndata*2/3)
training = shuffled_data[:ntraining]
testing = shuffled_data[ntraining:]

dt = DecisionTree(training)
print(dt.create_tree())

def test_mushrooms(test_set, verbose=False):
    ncorrect = 0
    ndeaths = 0
    n = len(test_set)
    for test in test_set:
        outcome = dt.get_decision(test)
        if verbose:
            print(dt.describe_outcome(outcome))
        if test[0] == outcome[0][0]:
            ncorrect += 1
        if test[0] == "p" and outcome[0][0] == "e":
            ndeaths += 1
    print("Out of {n}, {nc} were correct, for average of {a}\n"
          "Furthermore, there were {nd} cases where poisonous mushrooms "
          "were misidentified\n(likely meaning death for the eater...)"
          .format(n=n, nc=ncorrect, a=ncorrect/n, nd=ndeaths))
    
test_mushrooms(testing)



