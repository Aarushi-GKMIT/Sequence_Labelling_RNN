import json
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.utils import load_jsonl, build_token_vocab, build_label_vocab
from src.dataset import NERDataset
from src.model import RNNTagger
from src.evaluate import evaluate
from pathlib import Path

DATA_PATH = "data/dataset.jsonl"

EMBED_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 4
LR = 1e-3
EPOCHS = 15


def main():
    data = load_jsonl(DATA_PATH)

    token2id = build_token_vocab(data)
    label2id = build_label_vocab(data)

    max_len = max(len(row["tokens"]) for row in data)

    with open("src/artifacts/token2id.json", "w") as f:
        json.dump(token2id, f)

    with open("src/artifacts/label2id.json", "w") as f:
        json.dump(label2id, f)

    train_data, val_data = train_test_split(
        data, test_size=0.2, random_state=42
    )

    train_ds = NERDataset(train_data, token2id, label2id, max_len)
    val_ds = NERDataset(val_data, token2id, label2id, max_len)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = RNNTagger(
        vocab_size=len(token2id),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_labels=len(label2id),
        pad_idx=token2id["<PAD>"]
    )

    criterion = torch.nn.CrossEntropyLoss(ignore_index=label2id["O"])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for tokens, labels in train_loader:
            optimizer.zero_grad()
            logits = model(tokens)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        p, r, f1 = evaluate(model, val_loader, label2id)

        print(
            f"Epoch {epoch+1} | "
            f"Loss: {total_loss:.4f} | "
            f"P: {p:.3f} R: {r:.3f} F1: {f1:.3f}"
        )

   

    Path("artifacts").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "artifacts/model.pt")

    print("Model saved.")


if __name__ == "__main__":
    main()
