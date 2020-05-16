import torch
from torch import nn
import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np


class LinearModel(nn.Module):
    def __init__(self,inputD,outD,hidden=64):
        super(LinearModel, self).__init__()
        self.outfc = nn.Sequential(
            nn.Linear(inputD, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, outD),
        )

    def forward(self,x):
        return self.outfc(x)


































