import torch
import torch.nn as nn


class BOW(nn.Module):
    """A simple bag-of-words model"""

    def __init__(self, vocab_size, embedding_dim, vocab):
        super(BOW, self).__init__()
        self.vocab = vocab

        # this is a trainable look-up table with word embeddings
        self.embed = nn.Embedding(vocab_size, embedding_dim)

        # this is a trainable bias term
        self.bias = nn.Parameter(torch.zeros(embedding_dim), requires_grad=True)

    def forward(self, inputs):
        # this is the forward pass of the neural network
        # it applies a function to the input and returns the output

        # this looks up the embeddings for each word ID in inputs
        # the result is a sequence of word embeddings
        embeds = self.embed(inputs)

        # the output is the sum across the time dimension (1)
        # with the bias term added
        logits = embeds.sum(1) + self.bias

        return logits


class CBOW(nn.Module):
    """ A simple bag - of - words model """

    def __init__(self, vocab_size, embedding_dim, n_classes, vocab):
        super(CBOW, self).__init__()
        self.vocab = vocab

        # this is a trainable look-up table with word embeddings
        self.embed = nn.Embedding(vocab_size, embedding_dim)

        # this is a trainable bias term
        self.bias = nn.Parameter(torch.zeros(embedding_dim), requires_grad=True)

        # projection to n_classes
        self.project = nn.Linear(in_features=embedding_dim, out_features=n_classes)

    def forward(self, inputs):
        # this is the forward pass of the neural network
        # it applies a function to the input and returns the output

        # this looks up the embeddings for each word ID in inputs
        # the result is a sequence of word embeddings
        embeds = self.embed(inputs)

        # the output is the sum across the time dimension (1)
        # with the bias term added
        embed = embeds.sum(1) + self.bias

        logits = self.project(embed)

        return logits


class DeepCBOW(nn.Module):
    """A simple bag - of - words model"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_classes, vocab):
        super(DeepCBOW, self).__init__()
        self.vocab = vocab

        # this is a trainable look-up table with word embeddings
        self.embed = nn.Embedding(vocab_size, embedding_dim)

        # this is a trainable bias term
        self.bias = nn.Parameter(torch.zeros(embedding_dim), requires_grad=True)

        # feed-forward network
        self.ffn = nn.Sequential(nn.Linear(embedding_dim, hidden_dim),
                                 nn.Tanh(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.Tanh(),
                                 nn.Linear(hidden_dim, n_classes))

    def forward(self, inputs):
        # this is the forward pass of the neural network
        # it applies a function to the input and returns the output

        # this looks up the embeddings for each word ID in inputs
        # the result is a sequence of word embeddings
        embeds = self.embed(inputs)

        # the output is the sum across the time dimension (1)
        # with the bias term added
        embed = embeds.sum(1) + self.bias

        logits = self.ffn(embed)

        return logits


class PTDeepCBOW(DeepCBOW):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, vocab, vectors):
        super(PTDeepCBOW, self).__init__(
            vocab_size, embedding_dim, hidden_dim, output_dim, vocab)
        # copy the pre-trained embeddings
        self.embed.weight.data.copy_(torch.from_numpy(vectors))
        # disable training the pre-trained embeddings
        self.embed.weight.requires_grad = False