import torch.nn as nn


__all__ = ['mlp']


class MLP(nn.Module):

    def __init__(self, glorot_init, num_classes=10):
        super(MLP, self).__init__()
        self.FC1 = nn.Sequential(
            nn.Linear(3 * 32 * 32, 512),
            nn.ReLU(inplace=True)
        )
        self.FC2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True)
        )
        self.FC3 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(512, num_classes)
        if glorot_init:
            lins = [self.FC1[0], self.FC2[0], self.FC3[0], self.classifier]
            for l in lins:
                nn.init.xavier_uniform_(l.weight)
                nn.init.zeros_(l.bias)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = self.FC3(self.FC2(self.FC1(x)))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def mlp(**kwargs):
    model = MLP(**kwargs)
    return model
