import torch.nn as nn


__all__ = ['mlp']


class MLP(nn.Module):

    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(3 * 32 * 32, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def mlp(**kwargs):
    model = MLP(**kwargs)
    return model