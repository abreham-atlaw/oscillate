import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    def __init__(self, emb_size, ff_size):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(emb_size, ff_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(ff_size, emb_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out
