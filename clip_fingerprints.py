########################################################## Utils #######################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

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

from dataset import setup_dataloaders
from utils import *
from networks import *

from rdkit import Chem
from rdkit.Chem import AllChem

from x_clip import CLIP

from dataset import setup_dataloaders
from utils import save_img_as_npz, create_dir

################################################################## GPU PROCESS ###############################################################

# YOU CAN IGNORE THIS, IT IS JUST TO GET AVAILABLE GPU WHEN WORKING ON A SERVER

import subprocess
import sys
import torch

if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO


# check whether gpu is available on local system, else resort to CPU
def check_gpu():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU is available.')
        free_gpu_id = int(get_free_gpu())  # trouve le gpu libre grace a la fonction precedente
        torch.cuda.set_device(free_gpu_id)  # definie le gpu libre trouvee comme gpu de defaut pour PyTorch
        set_gpu = "cuda:" + str(free_gpu_id)
    else:
        device = torch.device('cpu')
        print("GPU not available")

    return device


def get_free_gpu():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv",
                                         "--query-gpu=memory.used,memory.free"])
    try:
        str_gpu_stats = StringIO(gpu_stats)
    except:
        str_gpu_stats = StringIO(gpu_stats.decode("utf-8"))
    gpu_df = pd.read_csv(str_gpu_stats,
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory.used'] = gpu_df['memory.used'].map(lambda x:
                                                      int(x.rstrip(' [MiB]')))
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x:
                                                      int(x.rstrip(' [MiB]')))
    idx = (gpu_df['memory.free'] - gpu_df['memory.used']).idxmax()
    print('Returning GPU{} with {} used MiB and {} free MiB'.format(idx,
                                                                    gpu_df.iloc[idx]['memory.used'],
                                                                    gpu_df.iloc[idx]['memory.free']))
    return idx


################################### Arguments ######################

def setup_args(device):
    options = argparse.ArgumentParser()
    options.add_argument('--datadir', action="store",
                         default="E:/Aaron/PhD/multimodal-drug-discovery/data/samples/")  # CAREFUL, HERE PUT THE PATH
    # OF WHERE IMAGES ARE STORED FOR YOU
    options.add_argument('--train-metafile', action="store", default="data/metadata/df00_train.csv")
    options.add_argument('--val-metafile', action="store", default="data/metadata/df00_test_easy.csv")
    options.add_argument('--val-hard-metafile', action="store", default="data/metadata/df00_test_hard.csv")
    options.add_argument('--dataset', action="store", default="cell-painting")

    options.add_argument('--featfile', action="store", default=None)
    options.add_argument('--img-size', action="store", default=512, type=int)

    options.add_argument('--n_sample', default=30, type=int, help='number of samples')
    options.add_argument('--seed', action="store", default=42, type=int)
    options.add_argument('--batch-size-train', action="store", dest="batch_size_train", default=32, type=int)
    options.add_argument('--batch-size-val', action="store", dest="batch_size_val", default=1, type=int)
    options.add_argument('--batch-size-val-hard', action="store", dest="batch_size_val_hard", default=1, type=int)
    if device == "cuda":
        options.add_argument('--num-workers', action="store", dest="num_workers", default=32, type=int)
    else:
        options.add_argument('--num-workers', action="store", dest="num_workers", default=0, type=int)

    # gpu options
    options.add_argument('--use-gpu', action="store_false", default=True)

    options.add_argument('--save_dir', action="store", default='test_save')

    # debugging mode
    options.add_argument('--debug-mode', action="store_true", default=False)

    options.add_argument('--use_nce_loss', action="store", default=False)

    return options.parse_args()


device = check_gpu()
args = setup_args(device)

############################################################ Fingerprints test #####################################

image_encoder = Image_Encoder()
mol_encoder = Mol_encoder_fingerprints()

clip = CLIP(
    image_encoder=image_encoder,
    text_encoder=mol_encoder,
    text_encode_without_mask=True,
    visual_image_size=512,
    dim_image=512,
    dim_text=512,
    dim_latent=512,
    channels=5
).to(device)
args.dataset = 'cell-painting'

trainloader, testloader, testloaderhard = setup_dataloaders(args)

print(len(trainloader))
print(len(testloader))
print(len(testloaderhard))

epochs = 2
learning_rate = 3e-4
optimizer = optim.Adam(clip.parameters(), lr=learning_rate, amsgrad=False)
scheduler = StepLR(optimizer, step_size=4000, gamma=0.5)

clip_epoch = []
time_epoch_liste = []
for epoch in range(epochs):
    print(epoch)
    time_epoch_start = time.time()
    clip.train()
    clip_error_batch = 0
    count = 0
    for batch_idx, (real_sample, cond) in enumerate(trainloader):
        real_sample = real_sample.to(device)
        ecfp = get_ecfp_tensor(cond)
        ecfp = ecfp.to(device)

        optimizer.zero_grad()
        loss = clip(ecfp, real_sample, return_loss=True)
        loss.backward()
        optimizer.step()

        clip_error_batch = loss.item() + clip_error_batch
        count = count + 1

    time_epoch_end = time.time()
    time_epoch = time_epoch_end - time_epoch_start
    time_epoch_liste.append(time_epoch)
    print('time for one epoch: ', time_epoch)
    print('approx time for one batch: ', time_epoch / len(trainloader))
    clip_epoch.append(clip_error_batch / count)

    print('%d iterations' % (epoch + 1))
    print('clip: %.3f' % np.mean(clip_epoch[-1:]))

    with open('clip_epoch.txt', 'w+') as f:
        for item in clip_epoch:
            f.write("%s " % item)

print(np.mean(time_epoch_liste))
