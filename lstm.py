import torch
import torch.nn as nn
import math
from prepare_data import load_data
from vocabulary import build_sentiment_mappings, build_pt_embeddings
from run_experiments import run_experiments


class MyLSTMCell(nn.Module):
    # LSTM
    def __init__(self, input_size, hidden_size, bias=True):
        """Creates the weights for this LSTM"""
        super(MyLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # YOUR CODE HERE
        self.proj_size = input_size+hidden_size
        self.transform = nn.Linear(self.proj_size, 4*hidden_size, bias=bias)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        """This is PyTorch's default initialization method"""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_, hx, mask=None):
        """
        input is (batch, input_size)
        hx is ((batch, hidden_size), (batch, hidden_size))
        """
        prev_h, prev_c = hx

        # project input and prev state
        proj = torch.cat((input_, prev_h), dim=1)

        # main LSTM computation
        states = self.transform(proj)
        i, f, g, o = torch.split(states, self.hidden_size, 1)
        i = self.sigmoid(i)
        f = self.sigmoid(f)
        g = self.tanh(g)
        o = self.sigmoid(o)

        c = f * prev_c + i * g
        h = o * self.tanh(c)

        return h, c

    def __repr__(self):
        return "{}({:d}, {:d})".format(
            self.__class__.__name__, self.input_size, self.hidden_size)


class LSTMClassifier(nn.Module):
    """Encodes sentence with an LSTM and projects final hidden state"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, vocab, vectors):
        super(LSTMClassifier, self).__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        with torch.no_grad():
            self.embed.weight.data.copy_(torch.from_numpy(vectors))
            self.embed.weight.requires_grad = False

        self.rnn = MyLSTMCell(embedding_dim, hidden_dim)

        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.5),  # explained later
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):

        B = x.size(0)  # batch size (this is 1 for now, i.e. 1 single example)
        T = x.size(1)  # timesteps (the number of words in the sentence)

        input_ = self.embed(x)

        # here we create initial hidden states containing zeros
        # we use a trick here so that, if input is on the GPU, then so are hx and cx
        hx = input_.new_zeros(B, self.rnn.hidden_size)
        cx = input_.new_zeros(B, self.rnn.hidden_size)

        # process input sentences one word/timestep at a time
        # input is batch-major (i.e., batch size is the first dimension)
        # so the first word(s) is (are) input_[:, 0]
        outputs = []
        for i in range(T):
            hx, cx = self.rnn(input_[:, i], (hx, cx))
            outputs.append(hx)

        # if we have a single example, our final LSTM state is the last hx
        if B == 1:
            final = hx
        else:
            #
            # This part is explained in next section, ignore this else-block for now.
            #
            # We processed sentences with different lengths, so some of the sentences
            # had already finished and we have been adding padding inputs to hx.
            # We select the final state based on the length of each sentence.

            # two lines below not needed if using LSTM from pytorch
            outputs = torch.stack(outputs, dim=0)  # [T, B, D]
            outputs = outputs.transpose(0, 1).contiguous()  # [B, T, D]

            # to be super-sure we're not accidentally indexing the wrong state
            # we zero out positions that are invalid
            pad_positions = (x == 1).unsqueeze(-1)

            outputs = outputs.contiguous()
            outputs = outputs.masked_fill_(pad_positions, 0.)

            mask = (x != 1)  # true for valid positions [B, T]
            lengths = mask.sum(dim=1)  # [B, 1]

            indexes = (lengths - 1) + torch.arange(B, device=x.device, dtype=x.dtype) * T
            final = outputs.view(-1, self.hidden_dim)[indexes]  # [B, D]

        # we use the last hidden state to classify the sentence
        logits = self.output_layer(final)
        return logits


class LSTMClassifier_Glove(LSTMClassifier):
    pass


class LSTMClassifier_W2V(LSTMClassifier):
    pass


def build_LSTM_models():
    v_glove, vectors_glove = build_pt_embeddings('glove')
    v_w2v, vectors_w2v = build_pt_embeddings('w2v')
    i2t, t2i = build_sentiment_mappings()

    models_fns = [
        LSTMClassifier_Glove,
        LSTMClassifier_W2V,
    ]

    models_args = [
        (len(v_glove.w2i), 300, 100, len(t2i), v_glove, vectors_glove),
        (len(v_w2v.w2i), 300, 100, len(t2i), v_w2v, vectors_w2v)
    ]

    nums_iterations = [
        50000,
        50000
    ]

    return models_fns, models_args, nums_iterations


if __name__ == "__main__":
    train_data, dev_data, test_data = load_data()
    seeds = ['42', '420', '4200']
    for seed in seeds:
        models_fns, models_args, nums_iterations = build_LSTM_models()
        run_experiments(models_fns, models_args, train_data, dev_data, test_data, nums_iterations,
                        base_name='LSTM', seed=seed)
