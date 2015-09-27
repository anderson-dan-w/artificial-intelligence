#!/usr/bin/python3
import random

import tfidf_setup as TFIDF
import neural_networks as NN

def train_nn(train_posts, max_iters=100):
    data_set = []
    for post in train_posts:
        post.get_tag_sim(TFIDF.TAGS)
        for tag in post.tags:
            title_cos, text_cos = post.tag_sim[tag]
            ## '1' since positive tag
            data_set.append([1, title_cos, text_cos])
        nontags = [tag for tag in TFIDF.TAGS if tag not in post.tags]
        random.shuffle(nontags)
        for i in range(5):
            title_cos, text_cos = post.tag_sim[nontags[i]]
            ## '0' since tag not applied to post
            data_set.append([0, title_cos, text_cos])
    print("Done getting data")
    nn = NN.NeuralNetwork(data_set)
    nn.online_training(max_iters=max_iters, debug=False)
    return nn

nn = train_nn(TFIDF.training_posts[:50])
