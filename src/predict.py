import torch
import json
from src.model import RNNTagger
from src.utils import pad_sequence

def load_artifacts():
    with open("src/artifacts/token2id.json") as f:
        token2id = json.load(f)
    with open("src/artifacts/label2id.json") as f:
        label2id = json.load(f)

    id2label = {v: k for k, v in label2id.items()}
    return token2id, label2id, id2label


def predict(tokens):
    token2id, label2id, id2label = load_artifacts()

    token_ids = [
        token2id.get(t, token2id["<UNK>"])
        for t in tokens
    ]

    token_ids = pad_sequence(
        token_ids,
        len(token_ids),
        token2id["<PAD>"]
    )

    x = torch.tensor([token_ids])

    model = RNNTagger(
        vocab_size=len(token2id),
        embed_dim=128,
        hidden_dim=256,
        num_labels=len(label2id),
        pad_idx=token2id["<PAD>"]
    )

    model.load_state_dict(torch.load("artifacts/model.pt"))
    model.eval()

    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(-1).squeeze(0)

    return [id2label[preds[i].item()] for i in range(len(tokens))]


if __name__ == "__main__":
   
    tokens = input("Enter the token sequence")
    labels = predict(tokens)

    for t, l in zip(tokens, labels):
        print(f"{t:15} → {l}")
