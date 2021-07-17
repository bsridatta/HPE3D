import torch.nn as nn


class LBAD(nn.Module):
    def __init__(self, neurons, activation, drop_out_p):
        super(LBAD, self).__init__()
        self.neurons = neurons
        self.activ = activation()
        self.drop_out_p = drop_out_p

        self.w1 = nn.Linear(self.neurons, self.neurons)
        self.bn1 = nn.BatchNorm1d(self.neurons)
        self.dropout = nn.Dropout(p=self.drop_out_p)

    def forward(self, x):
        x = self.w1(x)
        # x = self.bn1(x)
        x = self.activ(x)
        x = self.dropout(x)

        return x


class Critic(nn.Module):

    def __init__(self, neurons=1024, n_joints=15, activation=nn.LeakyReLU, drop_out_p=0.5):
        super(Critic, self).__init__()
        self.activation = activation
        self.neurons = neurons
        self.name = "critic"
        self.drop_out_p = drop_out_p
        self.n_joints = n_joints
        self.__build_model()

    def __build_model(self):

        self.inp_block = nn.Sequential(
            nn.Linear(2*self.n_joints, self.neurons), 
            self.activation(),
            # Shouldnt use BN for Critic input
        )

        self.LBAD_1 = LBAD(self.neurons, self.activation, self.drop_out_p)
        self.LBAD_2 = LBAD(self.neurons, self.activation, self.drop_out_p)

        self.LBAD_3 = LBAD(self.neurons, self.activation, self.drop_out_p)
        self.LBAD_4 = LBAD(self.neurons, self.activation, self.drop_out_p)
 
        self.out_block = nn.Sequential(
            nn.Linear(self.neurons, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if not x.is_contiguous():
            x = x.contiguous()
        x = x.view(-1, 2*self.n_joints)
        x = self.inp_block(x)

        residual = x
        x = self.LBAD_1(x)
        x = self.LBAD_2(x) + residual
        
        residual = x
        x = self.LBAD_3(x)
        x = self.LBAD_4(x) + residual
 
        out = self.out_block(x)
        
        return out
        