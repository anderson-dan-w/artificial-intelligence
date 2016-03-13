#!/usr/bin/env python3

from collections import defaultdict, Counter
import math
import random
from pprint import pprint
import time

import tfidf_setup as TFIDF
import ensemble as E
## do this after importing ensemble so that ens can load all the tags
TFIDF.TAGS.clean_post_tags(TFIDF.POSTS)

def get_sorted(tag_sim, n=0):
    return sorted(tag_sim.items(), key=lambda x:x[1][n])

def get_overlap(post, tag):
    title = [x for x in post.title_tf if x in TFIDF.TAGS.title_tfidf[tag]]
    text = [x for x in post.title_tf if x in TFIDF.TAGS.text_tfidf[tag]]
    return (title, text)

def test(test_posts, test_fn, n=5, DEBUG=False):
    tp, fp, fn = 0, 0, 0
    for idx, post in enumerate(test_posts):
        post.get_tag_sim(TFIDF.TITLE, TFIDF.TEXT, TFIDF.TAGS)
        top_tags = test_fn(post.tag_sim, n)
        true_positive = sum(1 for tag in top_tags if tag in post.tags)
        tp += true_positive
        fp += (len(top_tags) - true_positive)
        false_negs = [tag for tag in post.tags if tag not in top_tags]
        if false_negs and DEBUG:
            print("{} Missed {}/{}:{}".format(idx, len(false_negs),
                len(post.tags), false_negs))
        fn += len(false_negs)
    return tp, fp, fn

def top_title(tag_sim, n=5):
    ## returns [(tag, (title_cossim, text_cossim))]
    w_scores = sorted(tag_sim.items(), key=lambda x:x[1][0])[-1*n:]
    return set([t[0] for t in w_scores])

def top_text(tag_sim, n=5):
    w_scores = sorted(tag_sim.items(), key=lambda x:x[1][1])[-1*n:]
    return set([t[0] for t in w_scores])

def top_lin_title_text(tag_sim, n=5):
    w_scores = sorted(tag_sim.items(), key=lambda x:x[1][0] + x[1][1])[-1*n:]
    return set([t[0] for t in w_scores])

def top_nonlin_title_text(tag_sim, n=5):
    w_scores = sorted(tag_sim.items(), key=lambda x:x[1][0] * x[1][1])[-1*n:]
    return set([t[0] for t in w_scores])

def intersect_title_text(tag_sim, n=5):
    title_w_scores = sorted(tag_sim.items(), key=lambda x:x[1][0])[-2*n:]
    text_w_scores = sorted(tag_sim.items(), key=lambda x:x[1][1])[-2*n:]
    top10_title = set([t[0] for t in title_w_scores])
    top10_text = set([t[0] for t in text_w_scores])
    return top_title(tag_sim, 2*n).intersection(top_text(tag_sim, 2*n))

def union_title_text(tag_sim, n=5):
    return top_title(tag_sim, n).union(top_text(tag_sim, n))

def grid_search_weighted(test_posts, n=5):
    results = []
    for i in range(1, 10):
        title_weight = i/10
        text_weight = 1 - title_weight
        def weighted(tag_sim, n=n):
            tag_scores = {}
            for tag, value in tag_sim.items():
                title, text = value
                tag_scores[tag] = title*title_weight + text*text_weight
            return set(sorted(tag_scores, key=tag_scores.get)[-1*n:])
        tp, fp, fn = test(test_posts, weighted, n)
        results.append(tp/(tp+fn))
        #print("Weights {:.02}, {:.02} = {} {}".format(title_weight,
        #        text_weight, tp, fn))
    return results

def trivial_rnd(tag_sim, n=5):
    tags = [t for t in tag_sim]
    random.shuffle(tags)
    return set(tags[:n])

def trivial_common(tag_sim, n=5):
    return set(t[0] for t in TFIDF.TAGS.tag_counts.most_common(n))

funcs = [top_title, top_text, top_lin_title_text, top_nonlin_title_text,
        trivial_rnd, trivial_common]
def all_funcs(test_posts, funcs=funcs, n=5):
    results = []
    for func in funcs:
        tp, fp, fn = test(test_posts, func, n)
        #print("{: >22};n={} got: {:3} {:3}".format(func.__name__, n,tp, fn))
        results.append(tp/(tp+fn))
    return results

n=5
all5 = all_funcs(TFIDF.test_posts[-500:], n=n)
grid5 = grid_search_weighted(TFIDF.test_posts[-500:], n=n)
sall5 = "all,{},1,{},{}".format(TFIDF.chunk, n, ",".join("{:3.03}".format(v) for v in all5))
sgrid5 = "grid,{},1,{},{}".format(TFIDF.chunk, n, ",".join("{:3.03}".format(v) for v in [all5[0]] + grid5 + [all5[1]]))


n=10
all10 = all_funcs(TFIDF.test_posts[-500:], n=n)
grid10 = grid_search_weighted(TFIDF.test_posts[-500:], n=n)
sall10 = "all,{},1,{},{}".format(TFIDF.chunk, n, ",".join("{:3.03}".format(v) for v in all10))
sgrid10 = "grid,{},1,{},{}".format(TFIDF.chunk, n, ",".join("{:3.03}".format(v) for v in [all10[0]] + grid10 + [all10[1]]))

#R = open("results", "a")
fname = "results5"
fname=None
if fname:
    with open(fname, "a") as fh:
        for s in [sall5, sall10, sgrid5, sgrid10]:
            print(s, file=fh)
    for J in [3, 5, 20]:
        E.create_ensemble(J, fname)
        print("done w ensemble J =", J)

