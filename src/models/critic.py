import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, neurons=1024, activation=nn.ReLU, drop_out_p=0.2):
        super(LBAD, self).__init__()
        self.neurons = neurons
        self.activ = activation()
        self.drop_out_p = drop_out_p

        self.w1 = nn.Linear(self.neurons, self.neurons)
        self.bn1 = nn.BatchNorm1d(self.neurons)
        self.dropout = nn.Dropout(p=self.drop_out_p)
        self.name = "critic"

    def forward(self, x):
        x = self.w1(x)
        x = self.bn1(x)
        x = self.activ(x)
        x = self.dropout(x)

        return x