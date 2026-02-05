import torch
from sklearn.metrics import precision_recall_fscore_support

def evaluate(model, dataloader, label2id):
    model.eval()
    y_true, y_pred = [], []

    ignore_label = label2id["O"]

    with torch.no_grad():
        for tokens, labels in dataloader:
            logits = model(tokens)
            preds = logits.argmax(-1)

            for t_seq, p_seq in zip(labels, preds):
                for t, p in zip(t_seq, p_seq):
                    if t.item() == ignore_label:
                        continue
                    y_true.append(t.item())
                    y_pred.append(p.item())

    p, r, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )
    return p, r, f1

