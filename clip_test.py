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


from rdkit import Chem
from rdkit.Chem import AllChem

from dataset import setup_dataloaders
from utils import  save_img_as_npz,create_dir
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


################################################################## GPU PROCESS ###############################################################

# YOU CAN IGNORE THIS, IT IS JUST TO GET AVAILABLE GPU WHEN WORKING ON A SERVER

import subprocess
import sys
import torch
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO


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
    idx = (gpu_df['memory.free']-gpu_df['memory.used']).idxmax()
    print('Returning GPU{} with {} used MiB and {} free MiB'.format(idx,
gpu_df.iloc[idx]['memory.used'], gpu_df.iloc[idx]['memory.free']))
    return idx

free_gpu_id = int(get_free_gpu()) # trouve le gpu libre grace a la fonction precedente
torch.cuda.set_device(free_gpu_id) # definie le gpu libre trouvee comme gpu de defaut pour PyTorch
set_gpu="cuda:"+str(free_gpu_id)


#######################################################  Models ###########################################################

class Image_Encoder(torch.nn.Module):
    #output size is 512
    def __init__(self):
        super(Image_Encoder, self).__init__()
        self.model_pre = models.resnet18(pretrained=False)
        self.base=nn.Sequential(*list(self.model_pre.children()))
        self.base[0]=nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet=self.base[:-1]

    def forward(self, x):
        out=self.resnet(x)
        return out


class Mol_encoder(nn.Module):
    # input size is 2048 for ecfp fingerprints
    def __init__(self):
        super(Mol_encoder,self).__init__()
        self.fc1 = nn.Linear(2048,1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024,512)
        
    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class ProjectionHead_image(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        return projected


class ProjectionHead_text(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
    def forward(self, x):
        projected = self.projection(x)
        return projected




class CLIPModel(nn.Module):
    def __init__(
        self,
        image_embedding=512,
        text_embedding=512,
        projection_dim=512,
    ):
        super().__init__()
        self.image_encoder = Image_Encoder()
        self.mol_encoder = Mol_encoder()
        self.image_projection = ProjectionHead_image(image_embedding,projection_dim)
        self.mol_projection = ProjectionHead_text(text_embedding,projection_dim)
        self.image_embedding_size=image_embedding
        self.temperature = nn.Parameter(torch.tensor(0.2))

    def forward(self, real_sample, ecfp): 


        b, device = real_sample.shape[0], real_sample.device

        for p in self.mol_encoder.parameters():
          p.requires_grad = True

        for p in self.image_encoder.parameters():
          p.requires_grad = True
   

        image_features = self.image_encoder(real_sample)
        image_features_flat = image_features.view(-1, self.image_embedding_size)


        mol_features = self.mol_encoder(ecfp)
    
        image_latents = self.image_projection(image_features_flat)
        mol_latents = self.mol_projection(mol_features)
        


        mol_latents, image_latents = map(lambda t: F.normalize(t, p = 2, dim = -1), (mol_latents, image_latents))
       
        temp = self.temperature.exp()
      
        
        
        
        sim = einsum('i d, j d -> i j', mol_latents, image_latents) * temp
      
        labels = torch.arange(b, device = device)
      
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
       

        return loss







################################################# Define args ###################################################


def setup_args():

    options = argparse.ArgumentParser()

    options.add_argument('--datadir', action="store", default="/projects/imagesets3/Cell_Painting_dataset/subset_bray/images00/") # CAREFUL, HERE PUT THE PATH 
    #OF WHERE IMAGES ARE STORED FOR YOU
    options.add_argument('--train-metafile', action="store", default="data/metadata/df00_train.csv")  
    options.add_argument('--val-metafile', action="store", default="data/metadata/df00_test_easy.csv")
    options.add_argument('--val-hard-metafile', action="store", default="data/metadata/df00_test_hard.csv")
    options.add_argument('--dataset', action="store", default="cell-painting")
    
    options.add_argument('--featfile', action="store", default=None)
    options.add_argument('--img-size', action="store", default=512, type=int)

    options.add_argument('--n_sample', default=30, type=int, help='number of samples')
    options.add_argument('--seed', action="store", default=42, type=int)
    options.add_argument('--batch-size', action="store", dest="batch_size", default=16, type=int)
    options.add_argument('--num-workers', action="store", dest="num_workers", default=32, type=int)

    # gpu options
    options.add_argument('--use-gpu', action="store_false", default=True)

    options.add_argument('--save_dir',action="store",default='test_save')

    # debugging mode
    options.add_argument('--debug-mode', action="store_true", default=False)

    options.add_argument('--use_nce_loss', action="store",default=False )

    return options.parse_args()


###################################### Define training loop #################################


def get_ecfp6_fingerprints(mols): 
    """
    Get ECFP6 fingerprints for a list of molecules which may include `None`s,
    gracefully handling `None` values by returning a `None` value in that 
    position. 
    """
    fps = []
    for mol in mols:
        if mol is None:
            fps.append(None)
        else:
            mol=Chem.MolFromSmiles(mol)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 5, nBits=2048)
            fp.ToBitString()
            fps.append(fp)
    fps=np.array(fps)
    return(fps)





args = setup_args()

trainloader , testloader,testloaderhard = setup_dataloaders(args)
print(len(trainloader))
print(len(testloader))
print(len(testloaderhard))



model=CLIPModel().to(device)
epochs= 10
learning_rate = 3e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
scheduler = StepLR(optimizer, step_size=4000, gamma=0.5) 


clip_epoch=[]
time_epoch_liste=[]
for epoch in range(epochs):
    time_epoch_start = time.time()
    model.train()
    clip_error_batch=0
    count=0

    for batch_idx, (real_sample, cond) in enumerate(trainloader): 
        
        
        
        real_sample=real_sample.to(device)

        # process the ecfp
        ecfp=get_ecfp6_fingerprints(cond)
        ecfp_tensor=torch.from_numpy(ecfp)
        ecfp_tensor=ecfp_tensor.float()
        ecfp_tensor=ecfp_tensor.to(device)
        ####



        optimizer.zero_grad()
        loss=model(real_sample,ecfp_tensor)
        loss.backward()
        optimizer.step()

        clip_error_batch=loss.item()+clip_error_batch
        count=count+1

    
    
    time_epoch_end=time.time()
    time_epoch=time_epoch_end-time_epoch_start
    time_epoch_liste.append(time_epoch)
    print('time for one epoch: ',time_epoch)
    print('approx time for one batch: ',time_epoch/len(trainloader))
    clip_epoch.append(clip_error_batch/count)

    print('%d iterations' % (epoch+1))
    print('clip: %.3f' % np.mean(clip_epoch[-1:]))


print(np.mean(time_epoch_liste))

################################################## Test on easy task a first multimodal evaluation pipeline on Image retrieval task ###########################
print(len(testloader))
model.eval()

with torch.no_grad():
    mol_encoder=model.mol_encoder
    mol_proj=model.mol_projection
    image_encoder=model.image_encoder
    image_proj=model.image_projection

    #get the list of all molecules
    list_all_smiles=list(metadata_val['SMILES'])
    # get the fingerprints of all molecules
    ecfp=get_ecfp6_fingerprints(list_all_smiles)
    ecfp_tensor=torch.from_numpy(ecfp)
    ecfp_tensor=ecfp_tensor.float()
    ecfp_tensor=ecfp_tensor.to(device)
    
    # get the encoding of all molecules
    mol_features=mol_encoder(ecfp_tensor)
    mol_encoding=mol_proj(mol_features) # to get m1......mn


    mol_encoding_indices=list(range(mol_encoding.shape[0]))

    temperature=torch.tensor(0.2)

    top1acc_liste=[]
    top5acc_liste=[]
    top10acc_liste=[]

    for batch_idx, (real_sample, cond) in enumerate(testloader): 
        top1acc=0
        top5acc=0
        top10acc=0
        real_sample=real_sample.to(device)
        smiles_to_find=cond
        
        

        for i in range(len(list_all_smiles)):
                if smiles_to_find[0] == list_all_smiles[i]: #have to put 0 because it is a list too
                    ind_true=i
                   
                

        # encode image
        enc=image_encoder(real_sample)
        enc=enc.view(-1,enc.shape[1])
        image_feat=image_proj(enc)

        mol_latents, image_latents = map(lambda t: F.normalize(t, p = 2, dim = -1), (mol_encoding, image_feat))
        temp = temperature.exp()
        sim = einsum('i d, j d -> i j', mol_latents, image_latents) * temp
        probs=sim.softmax(dim=0).squeeze().detach().cpu()
          

        # compute top x%
        _,indices=torch.topk(probs,1)[0],torch.topk(probs,1)[1]
        indices=indices.cpu().numpy()
        if ind_true in indices:
            top1acc+=1

        _,indices=torch.topk(probs,5)[0],torch.topk(probs,5)[1]
        indices=indices.cpu().numpy()
        if ind_true in indices:
            top5acc+=1

        _,indices=torch.topk(probs,10)[0],torch.topk(probs,10)[1]
        indices=indices.cpu().numpy()
        if ind_true in indices:
            top10acc+=1


        top1acc_liste.append(top1acc)
        top5acc_liste.append(top5acc)
        top10acc_liste.append(top10acc)



# save all

final_top1acc=sum(top1acc_liste)/len(top1acc_liste)
final_top5acc=sum(top5acc_liste)/len(top5acc_liste)
final_top10acc=sum(top10acc_liste)/len(top10acc_liste)
print('final_top1acc: ',final_top1acc) 
print('final_top5acc: ',final_top5acc)   
print('final_top10acc: ',final_top10acc)     





     







