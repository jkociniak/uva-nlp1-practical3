from collections import Counter, OrderedDict
import numpy as np

"""#### Vocabulary
A first step in most NLP tasks is collecting all the word types that appear in the data into a vocabulary, and counting the frequency of their occurrences. On the one hand, this will give us an overview of the word distribution of the data set (what are the most frequent words, how many rare words are there, ...). On the other hand, we will also use the vocabulary to map each word to a unique numeric ID, which is a more handy index than a string.
"""
# Here we first define a class that can map a word to an ID (w2i)
# and back (i2w).


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first seen"""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__,
                           OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class Vocabulary:
    """A vocabulary, assigns IDs to tokens"""

    def __init__(self):
        self.freqs = OrderedCounter()
        self.w2i = {}
        self.i2w = []

    def count_token(self, t):
        self.freqs[t] += 1

    def add_token(self, t):
        self.w2i[t] = len(self.w2i)
        self.i2w.append(t)

    def build(self, min_freq=0):
        '''
        min_freq: minimum number of occurrences for a word to be included
                  in the vocabulary
        '''
        """The vocabulary has by default an `<unk>` token and a `<pad>` token."""
        self.add_token("<unk>")  # reserve 0 for <unk> (unknown words)
        self.add_token("<pad>")  # reserve 1 for <pad> (discussed later)

        tok_freq = list(self.freqs.items())
        tok_freq.sort(key=lambda x: x[1], reverse=True)
        for tok, freq in tok_freq:
            if freq >= min_freq:
                self.add_token(tok)


# This process should be deterministic and should have the same result
# if run multiple times on the same data set.


def build_vocabulary(datasets):
    v = Vocabulary()
    for data_set in datasets:
        for ex in data_set:
            for token in ex.tokens:
                v.count_token(token)

    v.build()
    print("Vocabulary size:", len(v.w2i))

    """#### Sentiment label vocabulary"""

    return v


def build_sentiment_mappings():
    # Now let's map the sentiment labels 0-4 to a more readable form
    i2t = ["very negative", "negative", "neutral", "positive", "very positive"]

    # And let's also create the opposite mapping.
    # We won't use a Vocabulary for this (although we could), since the labels
    # are already numeric.
    t2i = OrderedDict({p: i for p, i in zip(i2t, range(len(i2t)))})

    return i2t, t2i


word2vec_path = 'data/googlenews.word2vec.300d.txt'
glove_path = 'data/glove.840B.300d.sst.txt'


def build_pt_embeddings(pretrained_model):
    assert pretrained_model in ['glove', 'w2v']
    path = glove_path if pretrained_model == 'glove' else word2vec_path
    v = Vocabulary()
    embedding_dim = 300
    unk_embedding = [0] * embedding_dim
    pad_embedding = [0] * embedding_dim
    vectors = [unk_embedding, pad_embedding]
    with open(path, 'r') as f:
        for line in f:
            word, *embedding_str = line.split()
            embedding = [float(token) for token in embedding_str]
            v.count_token(word)
            vectors.append(embedding)

    v.build()
    vectors = np.stack(vectors, axis=0)
    return v, vectors

