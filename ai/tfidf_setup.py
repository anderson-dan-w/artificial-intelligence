#!/usr/bin/env python3

from collections import defaultdict, Counter
import math
import random
import re
import time
from xml.sax.saxutils import unescape

text = open("stopwords").read().upper()[:-1]
stopwords = set(text.split(","))

DATA_DIR = "/home/dwanderson/EN746_ML/proj_data/"
LING_POSTS = DATA_DIR + "lang_data/Posts.xml"
CS_POSTS = DATA_DIR + "cs_data/Posts.xml"

class Post(object):
    def __init__(self, tags, title, text):
        self.tags = tags
        self.original_tags = set([t for t in self.tags]) ## deepcopy
        self.title = title
        self.title_tf = Counter([x for x in self.title.split()
                            if x not in stopwords])
        count = sum(self.title_tf.values())
        for word in self.title_tf:
            self.title_tf[word] /= count
        self.title_tfidf = {}
        self.text = text
        self.text_tf = Counter([x for x in  self.text.split()
                            if x not in stopwords])
        count = sum(self.text_tf.values())
        for word in self.text_tf:
            self.text_tf[word] /= count
        self.text_tfidf = {}
        self.tag_sim = {}

    def get_tfidf(self, TITLE, TEXT):
        for tword in self.title_tf:
            count = TITLE.doc_count[tword]
            self.title_tfidf[tword] = (self.title_tf[tword] * TITLE.idf[tword])
        for tword in self.text_tf:
            count = TEXT.doc_count[tword]
            self.text_tfidf[tword] = (self.text_tf[tword] * TEXT.idf[tword]) 

    def get_tag_sim(self, TITLE, TEXT, TAGS, evaluate=False):
        if self.tag_sim and not evaluate:
            return
        if (not self.title_tfidf) or evaluate:
            self.get_tfidf(TITLE, TEXT)
        for tag in TAGS.tag_counts:
            tag_title, tag_text = TAGS.title_tfidf[tag], TAGS.text_tfidf[tag]
            title_cos = cos_sim(self.title_tfidf, tag_title, TITLE.WORDS)
            text_cos = cos_sim(self.text_tfidf, tag_text, TEXT.WORDS)
            self.tag_sim[tag] = (title_cos, text_cos)

##========================================================================##
def cos_sim(tfidf, tag_tfidf, WORD_SET):
    numerator = sum(tfidf[tword] * tag_tfidf[tword] for tword in tfidf
                    if tword in tag_tfidf)
    denominator = (math.sqrt(sum(tfidf[tword]**2 for tword in tfidf
                       if tword in WORD_SET)) *
                   math.sqrt(sum(tag_tfidf[tword]**2 for tword in tfidf
                       if tword in tag_tfidf)))
    if not denominator:
        return 0.0
    return numerator / denominator

##========================================================================##
class Title(object):
    def __init__(self, posts=None):
        self.WORDS = set()
        self.doc_count = defaultdict(int)
        self.idf = defaultdict(int)
        self.ndocs = 0
        if posts is not None:
            self.load(posts)

    def load(self, posts):
        self.ndocs += len(posts)
        for post in posts:
            self.WORDS.update(post.title_tf)
            for tword in post.title_tf:
                self.doc_count[tword] += 1
        for tword in self.WORDS:
            self.idf[tword] = math.log(self.ndocs/(1 + self.doc_count[tword]),2)

##========================================================================##
class Text(object):
    def __init__(self, posts=None):
        self.WORDS = set()
        self.ndocs = 0
        self.doc_count = defaultdict(int)
        self.idf = defaultdict(int)
        if posts is not None:
            self.load(posts)

    def load(self, posts):
        self.ndocs += len(posts)
        for post in posts:
            self.WORDS.update(post.text_tf)
            for tword in post.text_tf:
                self.doc_count[tword] += 1
        for tword in self.WORDS:
            self.idf[tword] = math.log(self.ndocs/(1 + self.doc_count[tword]),2)

##========================================================================##
class Tags(object):
    def __init__(self, posts=None, titles=None, texts=None):
        self.tag_counts = Counter()
        self.title_tf = defaultdict(lambda: defaultdict(int))
        self.text_tf = defaultdict(lambda: defaultdict(int))
        self.title_tfidf = defaultdict(lambda: defaultdict(int))
        self.text_tfidf = defaultdict(lambda: defaultdict(int))
        if all(v is not None for v in (posts, titles, texts)):
            #print("Loading in {} posts to tags".format(len(posts)))
            self.load(posts, titles, texts)

    def load(self, posts, titles, texts):
        if not isinstance(titles, list):
            titles = [titles]
        if not isinstance(texts, list):
            texts = [texts]
        for post in posts:
            self.tag_counts.update(post.tags)
            for tag in post.tags:
                self.title_tf[tag].update(post.title_tf)
                self.text_tf[tag].update(post.text_tf)
        for tag, count in self.tag_counts.items():
            for tword in self.title_tf[tag]:
                rel_fq = self.title_tf[tag][tword] / count
                self.title_tf[tag][tword]  = rel_fq
                tfidf = (sum(rel_fq * title.idf[tword] for title in titles)
                            / len(titles))
                #self.title_tfidf[tag][tword] = rel_fq * title.idf[tword]
                self.title_tfidf[tag][tword] = tfidf
            for tword in self.text_tf[tag]:
                rel_fq = self.text_tf[tag][tword] / count
                self.text_tf[tag][tword] = rel_fq
                tfidf = sum(rel_fq*text.idf[tword] for text in texts)/len(texts)
                #self.text_tfidf[tag][tword] = rel_fq * text.idf[tword]
                self.text_tfidf[tag][tword] = tfidf
    
    def clean_post_tags(self, posts):
        self.tag_counts = Counter({k: v for k, v in self.tag_counts.items()
                                   if v >= 10})
        for post in posts:
            post.tags = set(tag for tag in post.tags if tag in self.tag_counts)

##========================================================================##
entities = {"&#xA;":" ", "&quot;":" "}
def read_in(fname):
    questions = []
    with open(fname) as fh:
        lines = str(fh.read()).split("\n")
    for line in lines:
        match = re.search('PostTypeId="1".*?Body="(.*?)".*?Title="(.*?)".*?Tags="(.*?)"', line)
        if match is not None:
            text = unescape(match.group(1), entities).upper()
            text, _ = re.subn("<.*?>", " ", text)
            text, _ = re.subn("[^A-Z]", " ", text)
            text, _ = re.subn(" +", " ", text)
            text, _ = re.subn("^ | $", "", text)
            title = unescape(match.group(2), entities).upper()
            title, _ = re.subn("[^A-Z]", " ", title)
            title, _ = re.subn(" +", " ", title)
            title = "".join(l for l in title if l.isalpha() or l.isspace())
            tags = unescape(match.group(3), entities).upper()
            tags = set(tags[1:-1].split("><")) ## eg: <tag1><tag2>
            questions.append(Post(tags, title, text))
    return questions

print("Getting posts readin", end='')
start = time.time()
POSTS = read_in(CS_POSTS)
print("... which took {}".format(time.time()-start))
## for debug:
random.seed(1)
random.shuffle(POSTS)

chunk = 1000
training_posts = POSTS[:chunk]
test_posts = POSTS[chunk:]

##=========================================================================##
print("creating title_idf", end='')
start = time.time()
TITLE = Title()
TITLE.load(training_posts)
print("... which took {}".format(time.time()-start))

print("creating text_idf", end='')
start = time.time()
TEXT = Text()
TEXT.load(training_posts)
print("... which took {}".format(time.time()-start))

print("creating tag_tf", end='')
start = time.time()
TAGS = Tags()
TAGS.load(training_posts, TITLE, TEXT)
#TAGS.clean_post_tags(POSTS)
print("... which took {}".format(time.time()-start))

print("chunksize=", chunk)
