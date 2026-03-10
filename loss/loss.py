import torch.nn as nn



class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.L1 = nn.L1Loss()

    def forward(self, xs, ys):
        L1 = 1.0 * self.L1(xs, ys)
        return L1


class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.myloss = MyLoss()

    def forward(self, output, gt):
        myloss = self.myloss(output, gt)
        return myloss
