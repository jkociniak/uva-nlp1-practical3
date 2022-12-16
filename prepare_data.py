# run the TreeLSTM for subtrees
import re
from collections import namedtuple
from nltk import Tree


# A simple way to define a class is using namedtuple.
Example = namedtuple("Example", ["tokens", "tree", "label", "transitions"])


def filereader(path):
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            yield line.strip().replace("\\", "")


def tokens_from_treestring(s):
    """extract the tokens from a sentiment tree"""
    return re.sub(r"\([0-9] |\)", "", s).split()


def transitions_from_treestring(s):
    s = re.sub("\([0-5] ([^)]+)\)", "0", s)
    s = re.sub("\)", " )", s)
    s = re.sub("\([0-4] ", "", s)
    s = re.sub("\([0-4] ", "", s)
    s = re.sub("\)", "1", s)
    return list(map(int, s.split()))


def extract_subtrees(line):
    subtrees = []
    lb_ids = []
    for i, c in enumerate(line):
        if c == '(':
            lb_ids.append(i)
        elif c == ')':
            assert lb_ids
            last_lb_id = lb_ids.pop()
            subtrees.append(line[last_lb_id:i + 1])

    assert not lb_ids
    return subtrees


def line2example(line):
    tokens = tokens_from_treestring(line)
    tree = Tree.fromstring(line)  # use NLTK's Tree
    label = int(line[1])
    trans = transitions_from_treestring(line)
    return Example(tokens=tokens, tree=tree, label=label, transitions=trans)


def examplereader(path, lower=False, subtrees=False):
    """Returns all examples in a file one by one."""
    for line in filereader(path):
        line = line.lower() if lower else line
        if subtrees:
            st_lines = extract_subtrees(line)
            for st_line in st_lines:
                yield line2example(st_line)
        else:
            yield line2example(line)


def load_data(subtrees=False):
    # Let's load the data into memory.
    LOWER = False  # we will keep the original casing
    train_data = list(examplereader("trees/train.txt", lower=LOWER, subtrees=subtrees))
    dev_data = list(examplereader("trees/dev.txt", lower=LOWER))
    test_data = list(examplereader("trees/test.txt", lower=LOWER))

    print("train", len(train_data))
    print("dev", len(dev_data))
    print("test", len(test_data))
    return train_data, dev_data, test_data
