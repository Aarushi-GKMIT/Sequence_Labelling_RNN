import torch.nn as nn

class RNNTagger(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx
        )

        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        emb = self.embedding(x)      
        out, _ = self.rnn(emb)      
        logits = self.classifier(out)  
        return logits
