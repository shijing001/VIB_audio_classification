import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from utils import cuda
import pdb
import time
from numbers import Number
import numpy as np

class ToyNet(nn.Module):

    def __init__(self, K, output_features=8):
        super(ToyNet, self).__init__()
        self.K = K
        self.encode = nn.Sequential(
            nn.Linear(40,128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 2*self.K))
        
        self.decode = nn.Sequential(
                nn.Linear(self.K, output_features))

    def forward(self, x, num_sample=1):
        if x.dim() > 2 : x = x.view(x.size(0),-1)
        #x_traincnn = torch.from_numpy(np.expand_dims(x, axis=2))
        statistics = self.encode(x)
        #statistics=torch.flatten(statistics,start_dim=0, end_dim=1)
        mu = statistics[:,:self.K]
        std = F.softplus(statistics[:,self.K:]-5,beta=1) +1.0e-8
        encoding = self.reparametrize_n(mu,std,num_sample)
        logit = self.decode(encoding)

        if num_sample == 1 : pass
        elif num_sample > 1 : logit = F.softmax(logit, dim=2).mean(0)

        return (mu, std), logit

    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))

        return mu + eps * std

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])


def xavier_init(ms):
    for m in ms :
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight,gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()
