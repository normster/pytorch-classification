import torch
import torch.nn as nn

__all__ = ['cryptodl', 'cryptodl3']

class add_pool2d(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=-1, stride=1):
        super(add_pool2d, self).__init__()
        if padding == -1:
            padding = (kernel_size - 1) // 2
        self.layer = nn.Conv2d(channels, channels, stride=stride, kernel_size=kernel_size, padding=padding, groups=channels)
        self.layer.weight.requires_grad = False
        self.layer.weight.data = torch.ones_like(self.layer.weight.data)
        self.layer.bias.requires_grad = False
        self.layer.bias.data = torch.zeros_like(self.layer.bias.data)

    def forward(self, x):
        return self.layer(x)


class CryptoDL(nn.Module):
    def __init__(self, num_classes=10):
        super(CryptoDL, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=3, stride=2, padding=1)
        self.add_pool = add_pool2d(20, kernel_size=3) 
        self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(50)
        self.fc1 = nn.Linear(3200, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.bn1(self.add_pool(self.conv1(x)))
        x = self.bn2(self.conv2(x) ** 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x) ** 2
        x = self.fc2(x)
        return x

class CryptoDL3(nn.Module):
    def __init__(self, num_classes=10):
        super(CryptoDL3, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.bn2 = nn.BatchNorm2d(96)
        self.bn3 = nn.BatchNorm2d(96)
        self.bn4 = nn.BatchNorm2d(192)
        self.bn5 = nn.BatchNorm2d(192)
        self.bn6 = nn.BatchNorm2d(192)
        self.bn7 = nn.BatchNorm2d(192)
        self.fc = nn.Linear(12288, 10)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x) ** 2)
        x = self.bn4(self.conv4(x))
        x = self.bn5(self.conv5(x) ** 2)
        x = self.bn6(self.conv6(x))
        x = self.bn7(self.conv7(x))
        x = self.conv8(x) ** 2
        x = self.fc(x.view(x.size(0), -1))
        return x


def cryptodl(**kwargs):
    model = CryptoDL(**kwargs)
    return model

def cryptodl3(**kwargs):
    model = CryptoDL3(**kwargs)
    return model
