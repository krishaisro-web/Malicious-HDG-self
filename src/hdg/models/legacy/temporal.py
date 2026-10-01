import torch.nn as nn

class TemporalCombiner(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, domain_seq):
        out, _ = self.gru(domain_seq)
        return out[:, -1, :]
