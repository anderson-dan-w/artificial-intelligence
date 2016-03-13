#pr10
from __future__ import print_function, division
import math
import collections
import random

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

fname = "pr9_mushroom_attributes.csv"
attrs, attr_names = read_in_definitions(fname)
for idx, attr in enumerate(attr_names): print(idx, attr, attrs[attr].items(), end="\n\n")
    
def read_in_data(fname):
    with open(fname) as fh:
        lines = fh.read().split("\n")
    data = [l.split(",") for l in lines]
    return data

fname = "pr9_mushroom_data.csv"
data_set = read_in_data(fname)
print(data_set[0])

shuffled_data = data_set[:] # first copy it...
random.shuffle(shuffled_data)
ndata = len(shuffled_data)
ntraining = int(ndata*2/3)
training = shuffled_data[:ntraining]
testing = shuffled_data[ntraining:]

ntotal = len(training) 
npois = sum(1 for d in training if d[0] == "p")
nedible = ntotal - npois # true here; obviously not when there are more output labels.

def make_probabilities(training):
    probabilities = {}
    for idx, attr in enumerate(attr_names):
        if attr == "safe-to-eat?":
            continue
        probabilities[attr] = {}
        for label in attrs[attr]:
            counts = collections.Counter(d[0] for d in training if d[idx] == label)
            for key in ("p", "e"):
                counts.setdefault(key, 1) ## +1 smoothing
            nlabel = sum(counts.values())
            probabilities[attr][label] = (nlabel, counts["p"], counts["e"])
    return probabilities

probs = make_probabilities(training)
print(probs["odor"])

def calc_probabilities(vector, probs=probs):
    if len(vector) == 22:
        vector = "?" + vector   ## place-holder if no poisonous/edible provided
    results = {"p": npois/ntotal, "e": nedible/ntotal}
    for idx, attr in enumerate(attr_names):
        if attr == "safe-to-eat?":
            continue
        label = vector[idx]
        results["p"] *= (probs[attr][label][1]/npois)
        results["e"] *= (probs[attr][label][2]/nedible)
    return results

result = calc_probabilities(testing[0])
print(result)

def do_test(testing, probs=probs):
    ncorrect = 0
    ndeaths = 0
    for vector in testing:
        results = calc_probabilities(vector, probs)
        outcome = max(results, key=results.get)
        if outcome == vector[0]:
            ncorrect += 1
        if outcome == "e" and vector[0] == "p":
            ndeaths += 1
    return ncorrect, ndeaths

ncorrect, ndeaths = do_test(testing)
print("Got {} correct, out of {}, and there were {} fatal misclassifications"
        .format(ncorrect, len(testing), ndeaths))

