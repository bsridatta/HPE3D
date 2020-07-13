
import torch.utils.data as data
from mpiinf_dataset import MPIINF
import sys
sys.path.append("..")
from dataset import H36M

class H36M17_MPII(data.Dataset):
    def __init__(self, protocol, split):
        self.split = split
        self.H36M = H36M17(protocol, split)
        self.MPII = MPII()
        self.num_h36m = len(self.H36M)
        self.num_mpii = len(self.MPII)
        print('Load %d H36M and %d MPII samples' % (self.num_h36m, self.num_mpii))

    def __getitem__(self, index):
        if index < self.num_mpii:
            return self.H36M[index]
        else:
            return self.MPII[index - self.num_mpii]

    def __len__(self):
        return self.num_mpii*2