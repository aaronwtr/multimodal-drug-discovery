########################################################## Utils #######################################


import pandas as pd
import numpy as np

import json
from tqdm import tqdm
from tqdm.auto import tqdm
import logging
import random
from random import randint
import os
import glob
import time
from collections import defaultdict, OrderedDict
import re
import gc
from random import randint
import argparse
import sys
import logging


import torch
from torch import nn, einsum
from torch.utils.data import Dataset, DataLoader
from torch import cuda
from torch import nn
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import Identity
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable



import torchvision
from torchvision import transforms
from torchvision import models
from torchvision.utils import make_grid
import torchvision.datasets as datasets
from torchvision.utils import make_grid
from einops import rearrange


########################################## Image encoder Resnet 18 ###############################
'''
class Image_Encoder(torch.nn.Module):
    def __init__(self):
        super(Image_Encoder, self).__init__()
        self.model_pre = models.resnet18(pretrained=False)
        self.base=nn.Sequential(*list(self.model_pre.children()))
        self.base[0]=nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet=self.base[:-2]

    def forward(self, x):
        out=self.resnet(x)
        out=rearrange(out, 'b d h w -> b (h w) d')
        return out
'''

class Image_Encoder(torch.nn.Module):
    #output size is 512
    def __init__(self):
        super(Image_Encoder, self).__init__()
        self.model_pre = models.resnet18(pretrained=False)
        self.base=nn.Sequential(*list(self.model_pre.children()))
        self.base[0]=nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet=self.base[:-1]

    def forward(self, x):
        out=self.resnet(x).squeeze()
        return out

################################## Features encoder when using fingerprints ###################

class Mol_encoder_fingerprints(torch.nn.Module):
    #output size is 512
    def __init__(self):
        super(Mol_encoder_fingerprints, self).__init__()
        self.model =nn.Sequential(
                nn.Linear(2048,1024),
                nn.ReLU(),
                nn.Linear(1024,512)
        )

    def forward(self, x):
        out=self.model(x)
        return out

'''

class Mol_encoder_fingerprints(torch.nn.Module):
    def __init__(self):
        super(Mol_encoder_fingerprints, self).__init__()
        self.model =nn.Sequential(
                nn.Linear(1,1024),
                nn.ReLU(),
                nn.Linear(1024,512)
        )

    def forward(self, x):
        out=self.model(x)
        return out


class Mol_encoder_fingerprints(torch.nn.Module):
    def __init__(self):
        super(Mol_encoder_fingerprints, self).__init__()
        self.model =nn.Sequential(
                nn.Linear(1,512)
                
        )

    def forward(self, x):
        out=self.model(x)
        return out
'''
