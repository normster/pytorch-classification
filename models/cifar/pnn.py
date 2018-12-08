import torch
import torch.nn as nn

__all__ = ['pnn']


class PLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(PLinear, self).__init__(in_features, out_features, bias)
        self.scale = nn.Parameter(torch.ones(out_features))
        self.weight.requires_grad = False
        if bias:
            self.bias.requires_grad = False

    def forward(self, input):
        output = input.matmul(self.weight.t()) * self.scale
        if self.bias is not None:
            output += torch.jit._unwrap_optional(self.bias)
        return output


class PNN(nn.Module):

    def __init__(self, glorot_init, num_classes=10):
        super(PNN, self).__init__()
        self.FC1 = nn.Sequential(
            nn.PLinear(3 * 32 * 32, 512),
            nn.ReLU(inplace=True)
        )
        self.FC2 = nn.Sequential(
            nn.PLinear(512, 512),
            nn.ReLU(inplace=True)
        )
        self.FC3 = nn.Sequential(
            nn.PLinear(512, 512),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.PLinear(512, num_classes)
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


def pnn(**kwargs):
    model = PNN(**kwargs)
    return model
