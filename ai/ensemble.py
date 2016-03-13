#!/usr/bin/env python3
from __future__ import print_function
from collections import defaultdict, Counter

import tfidf_setup as TFIDF

class Ensemble(object):
    def __init__(self, posts, nensembles=5):
        self.nensembles = nensembles
        self.posts = []
        self.titles = []
        self.texts = []
        self.tagsets = []
        nchunk = int(len(posts)/nensembles)
        for i in range(nensembles):
            self.posts.append([])
            for j in range(nchunk):
                p = posts[i*nchunk + j]
                post = TFIDF.Post(p.original_tags, p.title, p.text)
                self.posts[i].append(post)
            self.titles.append(TFIDF.Title(self.posts[i]))
            self.texts.append(TFIDF.Text(self.posts[i]))
            self.tagsets.append(TFIDF.Tags(self.posts[i],
                                    self.titles[i], self.texts[i]))

    def get_tag_sims(self, post):
        tag_sims = []
        for i in range(self.nensembles):
            ## True -> evaluate with args, don't fall back on precomputed
            post.get_tag_sim(self.titles[i], self.texts[i], self.tags[i], True)
            tag_sims.append(post.tag_sim.copy())
        return tag_sims

    def clean_post_tags(self, posts):
        aggregate = Counter()
        for tagset in self.tagsets:
            aggregate.update(tagset.tag_counts)
        aggregate = Counter({k : v for k,v in aggregate.items() if v >= 10})
        for idx, tagset in enumerate(self.tagsets):
            tagset_count = Counter({k:v for k, v in tagset.tag_counts.items() 
                                if k in aggregate})
            self.tagsets[idx].tag_counts = tagset_count
        for post in posts:
            post.tags = set(tag for tag in post.tags if tag in aggregate)

##=========================================================================##
class EnsemblePost(TFIDF.Post):
    def __init__(self, tags, title, text):
        super(EnsemblePost, self).__init__(tags, title, text)
        self.title_tfidf = []
        self.text_tfidf = []
        self.tag_sims = []

    def get_tfidf(self, ens):
        for i in range(ens.nensembles):
            TITLE, TEXT = ens.titles[i], ens.texts[i]
            title, text = {}, {}
            for tword in self.title_tf:
                count = TITLE.doc_count[tword]
                title[tword] = self.title_tf[tword] * TITLE.idf[tword]
            for tword in self.text_tf:
                count = TEXT.doc_count[tword]
                text[tword] = self.text_tf[tword] * TEXT.idf[tword]
            self.title_tfidf.append(title)
            self.text_tfidf.append(text)

    def get_tag_sim(self, ens, evaluate=False):
        if self.tag_sims and not evaluate:
            return
        if (not self.title_tfidf) or evaluate:
            self.get_tfidf(ens)
        for i in range(ens.nensembles):
            tag_sim = {}
            title, text = self.title_tfidf[i], self.text_tfidf[i]
            tagset = ens.tagsets[i]
            TITLE_W, TEXT_W = ens.titles[i].WORDS, ens.texts[i].WORDS
            for tag in tagset.tag_counts:
                tag_title, tag_text = tagset.title_tfidf[tag], tagset.text_tfidf[tag]
                title_cos = TFIDF.cos_sim(title, tag_title, TITLE_W)
                text_cos = TFIDF.cos_sim(text, tag_text, TEXT_W)
                tag_sim[tag] = (title_cos, text_cos)
            self.tag_sims.append(tag_sim)

##=========================================================================##
def top_title_ens(tag_sims, n=5):
    combo = defaultdict(lambda: [0, 0])
    for ts in tag_sims:
        for tag, vals in ts.items():
            combo[tag][0] += vals[0]
            combo[tag][1] += vals[1]
    ## normalize
    for tag, vals in combo.items():
        div = sum(1 for tagset in tag_sims if tag in tagset)
        combo[tag][0] = vals[0]/div
        combo[tag][1] = vals[1]/div
    topn = sorted(combo.items(), key=lambda x:x[1][0])[-1*n:]
    toptags = set(t[0] for t in topn)
    return toptags

def top_text_ens(tag_sims, n=5):
    combo = defaultdict(lambda: [0, 0])
    for ts in tag_sims:
        for tag, vals in ts.items():
            combo[tag][0] += vals[0]
            combo[tag][1] += vals[1]
    ## normalize
    for tag, vals in combo.items():
        div = sum(1 for tagset in tag_sims if tag in tagset)
        combo[tag][0] = vals[0]/div
        combo[tag][1] = vals[1]/div
    topn = sorted(combo.items(), key=lambda x:x[1][1])[-1*n:]
    toptags = set(t[0] for t in topn)
    return toptags

def top_lin_ens(tag_sims, n=5):
    combo = defaultdict(lambda: [0, 0])
    for ts in tag_sims:
        for tag, vals in ts.items():
            combo[tag][0] += vals[0]
            combo[tag][1] += vals[1]
    ## normalize
    for tag, vals in combo.items():
        div = sum(1 for tagset in tag_sims if tag in tagset)
        combo[tag][0] = vals[0]/div
        combo[tag][1] = vals[1]/div
    topn = sorted(combo.items(), key=lambda x:x[1][0] + x[1][1])[-1*n:]
    toptags = set(t[0] for t in topn)
    return toptags

def top_nonlin_ens(tag_sims, n=5):
    combo = defaultdict(lambda: [0, 0])
    for ts in tag_sims:
        for tag, vals in ts.items():
            combo[tag][0] += vals[0]
            combo[tag][1] += vals[1]
    ## normalize
    for tag, vals in combo.items():
        div = sum(1 for tagset in tag_sims if tag in tagset)
        combo[tag][0] = vals[0]/div
        combo[tag][1] = vals[1]/div
    topn = sorted(combo.items(), key=lambda x:x[1][0] * x[1][1])[-1*n:]
    toptags = set(t[0] for t in topn)
    return toptags

def intersect(tag_sims, n=5):
    return top_title_ens(tag_sims, 2*n).intersection(top_text_ens(tag_sims, 2*n))

def union_ens(tag_sims, n=5):
    return top_title_ens(tag_sims, n).union(top_text_ens(tag_sims, n))

def gridsearch_weighted(test_posts, n=5):
    results = []
    for i in range(1, 10):
        title_weight = i/10
        text_weight = 1 - title_weight
        def weighted(tag_sims, n=n):
            tag_scores = {}
            combo = defaultdict(lambda: [0, 0])
            for ts in tag_sims:
                for tag, vals in ts.items():
                    combo[tag][0] += vals[0]
                    combo[tag][1] += vals[1]
            ## normalize
            for tag, vals in combo.items():
                div = sum(1 for tagset in tag_sims if tag in tagset)
                combo[tag][0] = vals[0]/div
                combo[tag][1] = vals[1]/div
            for tag, value in combo.items():
                title, text = value
                tag_scores[tag] = title*title_weight + text*text_weight
            return set(sorted(tag_scores, key=tag_scores.get)[-1*n:])
        tp, fp, fn = ensemble_test(test_posts, weighted, n)
        #print("ensemble weights {:.02}, {:.02} = {} {}".format(title_weight,
        #    text_weight, tp, fn))
        results.append(tp/(tp+fn))
    return results

def ensemble_test(test_posts, test_fn, n=5):
    tp, fp, fn = 0, 0, 0
    for idx, post in enumerate(test_posts):
        tag_sims = post.tag_sims
        top_tags = test_fn(tag_sims, n=n)
        tp_count = sum(1 for tag in top_tags if tag in post.tags)
        tp += tp_count
        fp += (len(top_tags) - tp_count)
        fn += len([tag for tag in post.tags if tag not in top_tags])
    return tp, fp, fn

funcs = [top_title_ens, top_text_ens, top_lin_ens, top_nonlin_ens]
def all_funcs(test_posts, funcs=funcs, n=5):
    results = []
    for func in funcs:
        tp, fp, fn = ensemble_test(test_posts, func, n)
        #print("{: >15};n={} got: {:3} {:3}".format(func.__name__, n, tp, fn))
        results.append(tp/(tp+fn))
    return results

##=========================================================================##
EPOSTS = [EnsemblePost(p.tags, p.title, p.text) for p in TFIDF.POSTS]
chunk = TFIDF.chunk
training_posts = EPOSTS[:chunk]
test_posts = EPOSTS[chunk:]
#print("Done ensemble-post-readin")

def create_ensemble(J, fname=None):
    e = Ensemble(training_posts, J)
    e.clean_post_tags(EPOSTS)
    for i in range(500):
        test_posts[-1*i].get_tag_sim(e, True)
    for n in [5, 10]:
        allf = all_funcs(test_posts[-500:], n=n)
        grid = gridsearch_weighted(test_posts[-500:], n=n)
        sallf = "all,{},{},{},{}".format(chunk,J,n,",".join("{:3.03}".format(v) 
                    for v in allf))
        sgrid = "grid,{},{},{},{}".format(chunk,J,n,",".join("{:3.03}".format(v)
                    for v in [allf[0]] + grid + [allf[1]]))
        if fname:
            with open(fname, "a") as fh:
                print(sallf, file=fh)
                print(sgrid, file=fh)
