import torch
import torch.nn as nn
import numpy as np
import time
import random
from sklearn.metrics import f1_score, roc_auc_score


# Here we print each parameter name, shape, and if it is trainable.
def print_parameters(model):
    total = 0
    for name, p in model.named_parameters():
        total += np.prod(p.shape)
        print("{:24s} {:12s} requires_grad={}".format(name, str(list(p.shape)), p.requires_grad))
    print("\nTotal number of parameters: {}\n".format(total))


# def prepare_example(example, vocab, device):
#     """
#     Map tokens to their IDs for a single example
#     """
#
#     # vocab returns 0 if the word is not there (i2w[0] = <unk>)
#     x = [vocab.w2i.get(t, 0) for t in example.tokens]
#
#     x = torch.LongTensor([x])
#     x = x.to(device)
#
#     y = torch.LongTensor([example.label])
#     y = y.to(device)
#
#     return x, y
#
#
# def simple_evaluate(model, data, prep_fn=prepare_example, **kwargs):
#     """Accuracy of a model on given data set."""
#     correct = 0
#     total = 0
#     model.eval()  # disable dropout (explained later)
#
#     for example in data:
#         # convert the example input and label to PyTorch tensors
#         x, target = prep_fn(example, model.vocab)
#
#         # forward pass without backpropagation (no_grad)
#         # get the output from the neural network for input x
#         with torch.no_grad():
#             logits = model(x)
#
#         # get the prediction
#         prediction = logits.argmax(dim=-1)
#
#         # add the number of correct predictions to the total correct
#         correct += (prediction == target).sum().item()
#         total += 1
#
#     return correct, total, correct / float(total)
#
#
# def get_examples(data, shuffle=True, **kwargs):
#     """Shuffle data set and return 1 example at a time (until nothing left)"""
#     if shuffle:
#         print("Shuffling training data")
#         random.shuffle(data)  # shuffle training data each epoch
#     for example in data:
#         yield example


def get_minibatch(data, batch_size=25, shuffle=True):
    """Return minibatches, optional shuffling"""
    if shuffle:
        print("Shuffling training data")
        random.shuffle(data)  # shuffle training data each epoch

    batch = []

    # yield minibatches
    for example in data:
        batch.append(example)

        if len(batch) == batch_size:
            yield batch
            batch = []

    # in case there is something left
    if len(batch) > 0:
        yield batch


def pad(tokens, length, pad_value=1):
    """add padding 1s to a sequence to that it has the desired length"""
    return tokens + [pad_value] * (length - len(tokens))


def prepare_minibatch(mb, vocab, device):
    """
    Minibatch is a list of examples.
    This function converts words to IDs and returns
    torch tensors to be used as input/targets.
    """
    batch_size = len(mb)
    maxlen = max([len(ex.tokens) for ex in mb])

    # vocab returns 0 if the word is not there
    x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen) for ex in mb]

    x = torch.LongTensor(x)
    x = x.to(device)

    y = [ex.label for ex in mb]
    y = torch.LongTensor(y)
    y = y.to(device)

    return x, y


def train_eval(model, data, device,
               batch_fn=get_minibatch, prep_fn=prepare_minibatch,
               batch_size=16):
    """Accuracy of a model on given data set (using mini-batches)"""
    correct = 0
    total = 0
    model.eval()  # disable dropout

    for mb in batch_fn(data, batch_size=batch_size, shuffle=False):
        x, targets = prep_fn(mb, model.vocab, device)
        with torch.no_grad():
            logits = model(x)

        predictions = logits.argmax(dim=-1).view(-1)

        # probs = torch.softmax(logits, dim=-1)
        # f1 = f1_score(targets, predictions)
        # roc_auc = roc_auc_score(targets, predictions)

        # add the number of correct predictions to the total correct
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

    return correct, total, correct / float(total)


def test_eval(model, data, device,
              batch_fn=get_minibatch, prep_fn=prepare_minibatch,
              batch_size=16):
    """Accuracy of a model on given data set (using mini-batches)"""
    model.eval()  # disable dropout

    targets = []
    probs = []
    total = 0
    for mb in batch_fn(data, batch_size=batch_size, shuffle=False):
        x, targets_batch = prep_fn(mb, model.vocab, device)
        targets.append(targets_batch.cpu())
        with torch.no_grad():
            logits = model(x)

        probs_batch = logits.softmax(dim=-1)
        probs.append(probs_batch.cpu())

        total += targets_batch.size(0)
    probs = np.concatenate(probs)
    targets = np.concatenate(targets)

    roc_auc = roc_auc_score(targets, probs, multi_class='ovr', average='macro')
    predictions = probs.argmax(axis=-1)
    f1 = f1_score(targets, predictions, average='macro')
    correct = (predictions == targets).sum().item()
    accuracy = correct / float(total)

    return accuracy, f1, roc_auc


def check_correctness(input):
    test_fns = {
        'inf': lambda x: torch.isinf(x).any(),
        'nan': lambda x: torch.isnan(x).any()
    }
    for name, fn in test_fns.items():
        if fn(input):
            print(f'test for {name} failed')
            raise Exception


def train_model(model, optimizer,
                train_data, dev_data, test_data, device, seed,
                num_iterations=10000,
                print_every=1000, eval_every=1000,
                batch_fn=get_minibatch,
                prep_fn=prepare_minibatch,
                train_eval_fn=train_eval,
                test_eval_fn=test_eval,
                batch_size=1, eval_batch_size=None):
    """Train a model."""
    model.to(device)
    iter_i = 0
    train_loss = 0.
    print_num = 0
    start = time.time()
    criterion = nn.CrossEntropyLoss()  # loss function
    best_eval = 0.
    best_iter = 0

    # store train loss and validation accuracy during training
    # so we can plot them afterwards
    losses = []
    accuracies = []

    if eval_batch_size is None:
        eval_batch_size = batch_size

    while True:  # when we run out of examples, shuffle and continue
        for batch in batch_fn(train_data, batch_size=batch_size):

            # forward pass
            model.train()
            x, targets = prep_fn(batch, model.vocab, device)
            logits = model(x)
            check_correctness(logits)

            B = targets.size(0)  # later we will use B examples per update

            # compute cross-entropy loss (our criterion)
            # note that the cross entropy loss function computes the softmax for us
            loss = criterion(logits.view([B, -1]), targets.view(-1))
            check_correctness(loss)

            train_loss += loss / B

            # backward pass (tip: check the Introduction to PyTorch notebook)

            # erase previous gradients
            optimizer.zero_grad()

            # compute gradients
            loss.backward()

            # update weights - take a small step in the opposite dir of the gradient
            optimizer.step()

            print_num += 1
            iter_i += 1

            # print info
            if iter_i % print_every == 0:
                print("Iter %r: loss=%.4f, time=%.2fs" %
                      (iter_i, train_loss, time.time() - start))
                losses.append(train_loss.detach().numpy())
                print_num = 0
                train_loss = 0.

            # evaluate
            if iter_i % eval_every == 0:
                _, _, accuracy = train_eval_fn(model, dev_data, device, batch_size=eval_batch_size,
                                               batch_fn=batch_fn, prep_fn=prep_fn)
                accuracies.append(accuracy)
                print("iter %r: dev acc=%.4f" % (iter_i, accuracy))

                # save best model parameters
                if accuracy > best_eval:
                    print("new highscore")
                    best_eval = accuracy
                    best_iter = iter_i
                    path = "{}.pt".format(model.__class__.__name__)
                    ckpt = {
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_eval": best_eval,
                        "best_iter": best_iter
                    }
                    torch.save(ckpt, path)

            # done training
            if iter_i == num_iterations:
                print("Done training")

                # evaluate on train, dev, and test with best model
                print("Loading best model")
                path = "{}_best_seed={}.pt".format(model.__class__.__name__, seed)
                ckpt = torch.load(path)
                model.load_state_dict(ckpt["state_dict"])

                train_acc, train_f1, train_roc_auc = test_eval_fn(
                    model, train_data, device, batch_size=eval_batch_size,
                    batch_fn=batch_fn, prep_fn=prep_fn)
                dev_acc, dev_f1, dev_roc_auc = test_eval_fn(
                    model, dev_data, device, batch_size=eval_batch_size,
                    batch_fn=batch_fn, prep_fn=prep_fn)
                test_acc, test_f1, test_roc_auc = test_eval_fn(
                    model, test_data, device, batch_size=eval_batch_size,
                    batch_fn=batch_fn, prep_fn=prep_fn)

                print("best model iter {:d}: "
                      "train acc={:.4f}, dev acc={:.4f}, test acc={:.4f}".format(
                       best_iter, train_acc, dev_acc, test_acc))

                best_model_metrics = test_acc, test_f1, test_roc_auc

                return losses, accuracies, best_model_metrics


def prepare_treelstm_minibatch(mb, vocab, device):
    """
    Returns sentences reversed (last word first)
    Returns transitions together with the sentences.
    """
    batch_size = len(mb)
    maxlen = max([len(ex.tokens) for ex in mb])

    # vocab returns 0 if the word is not there
    # NOTE: reversed sequence!
    x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen)[::-1] for ex in mb]

    x = torch.LongTensor(x)
    x = x.to(device)

    y = [ex.label for ex in mb]
    y = torch.LongTensor(y)
    y = y.to(device)

    maxlen_t = max([len(ex.transitions) for ex in mb])
    transitions = [pad(ex.transitions, maxlen_t, pad_value=2) for ex in mb]
    transitions = np.array(transitions)
    transitions = transitions.T  # time-major

    return (x, transitions), y
