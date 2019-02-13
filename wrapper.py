import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Wrapper(nn.Module):

    def __init__(self, student, teacher):
        super(Wrapper, self).__init__()
        self.student = student
        self.teacher = teacher
        self.student.fc = copy.deepcopy(self.teacher.fc)


    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()


    def forward(self, x, mode='train'):
        if mode == 'eval':
            return self.student(x)
        else:
            tx = x.detach()
            layer_loss = 0
            for s, t in zip(self.student.network.children(), self.teacher.network.children()):
                sx = s(tx)
                tx = t(tx)

                layer_loss += F.mse_loss(sx, tx)
                tx = tx.detach()

            if mode == 'train':
                return self.student(x), layer_loss
            elif mode == 'train_no_student':
                return layer_loss
            else:
                raise Exception("mode not recognized")
