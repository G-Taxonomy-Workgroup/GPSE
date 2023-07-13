import torch.nn as nn


class IdentityHead(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, batch):
        return batch.x, batch.y
