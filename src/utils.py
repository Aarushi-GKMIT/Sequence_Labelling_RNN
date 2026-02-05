import json
from collections import Counter

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_token_vocab(dataset):
    counter = Counter()
    for row in dataset:
        counter.update(row["tokens"])

    token2id = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1
    }

    for token in counter:
        token2id[token] = len(token2id)

    return token2id


def build_label_vocab(dataset):
    labels = set()
    for row in dataset:
        labels.update(row["labels"])

    return {label: i for i, label in enumerate(sorted(labels))}


def encode(tokens, token2id):
    return [
        token2id.get(t, token2id[UNK_TOKEN])
        for t in tokens
    ]


def pad_sequence(seq, max_len, pad_value):
    return seq + [pad_value] * (max_len - len(seq))
