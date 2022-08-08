import torch
from torch import optim
import logging
import os
import json
import numpy as np
import random
from dataset import setup_dataloaders
from utils import save_img_as_npz,create_dir
import argparse
import sys
from torchvision.utils import save_image
import time


import pandas as pd


import torch
import transformers
from transformers import RobertaModel, RobertaTokenizer
from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline, RobertaModel, RobertaTokenizer
from transformers import RobertaForMaskedLM
from transformers import RobertaConfig
from torch import cuda
from torch.nn import Identity
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from rdkit.Chem import AllChem
from tokenizer import HugTokenizer


def setup_args():

    options = argparse.ArgumentParser()

    options.add_argument('--datadir', action="store", default="/projects/imagesets3/Cell_Painting_dataset/subset_bray/images00/")
    options.add_argument('--train-metafile', action="store", default="data/metadata/df00_train.csv")  
    options.add_argument('--val-metafile', action="store", default="data/metadata/df00_test_easy.csv")
    options.add_argument('--val-hard-metafile', action="store", default="data/metadata/df00_test_hard.csv")
    options.add_argument('--dataset', action="store", default="cell-painting")
    
    options.add_argument('--featfile', action="store", default=None)
    options.add_argument('--img-size', action="store", default=512, type=int)

    options.add_argument('--n_sample', default=30, type=int, help='number of samples')
    options.add_argument('--seed', action="store", default=42, type=int)
    options.add_argument('--batch-size-train', action="store", dest="batch_size_train", default=128, type=int)
    options.add_argument('--batch-size-val', action="store", dest="batch_size_val", default=1, type=int)
    options.add_argument('--batch-size-val-hard', action="store", dest="batch_size_val_hard", default=1, type=int)
    options.add_argument('--num-workers', action="store", dest="num_workers", default=60, type=int)

    # gpu options
    options.add_argument('--use-gpu', action="store_false", default=True)

    options.add_argument('--save_dir',action="store",default='test_save')

    # debugging mode
    options.add_argument('--debug-mode', action="store_true", default=False)

    options.add_argument('--use_nce_loss', action="store",default=False )

    return options.parse_args()


# to work or visualize only with 3 channels: the ones taken represents the actin, WGA/phalloidin and nuclei (in RGB order)
def get_3c_image(image):
  return torch.stack((image[:,3,:,:],image[:,1,:,:],image[:,4,:,:])).permute(1,0,2,3)

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

create_dir(args.save_dir)

if args.dataset == 'cell-painting':
  
  trainloader , testloader,testloaderhard = setup_dataloaders(args)
  print(len(trainloader))
  print(len(testloader))
  print(len(testloaderhard))
  time_epoch_start = time.time()
  
  for batch_idx, (real_sample, cond) in enumerate(trainloader): # real sample are the real images and cond are the molecules (for of list len(bs))
      print(batch_idx)
      #print(real_sample.shape)
      #print(len(cond))
      
      '''
      # to save images from the current batch in one channel per one channel
      for ch in range(5):
          ch_img = real_sample[:,ch:ch+1,:,:]
          save_image(ch_img, os.path.join(args.save_dir, 'examples%s_%s_real.png' % (batch_idx, ch)), 
                      normalize=True, nrow=10, range=(-0.5, 0.5))

      
      # to save images from the current batch in three channel visu
      ch_img = get_3c_image(real_sample)
      save_image(ch_img, os.path.join(args.save_dir, 'examples%s_%s_real_3channel.png' % (batch_idx, ch)), 
                  normalize=True, nrow=10, range=(-0.5, 0.5))

      break
      '''
  time_epoch_end=time.time()
  time_epoch=time_epoch_end-time_epoch_start
  print('time for one epoch: ',time_epoch)
  print('approx time for one batch: ',time_epoch/len(trainloader))
  

      

elif args.dataset == 'cell-painting-smiles':

  tokenizer = HugTokenizer("SMILES")
  args.tokenizer=tokenizer
  
  trainloader , testloader,testloaderhard = setup_dataloaders(args)
  print(len(trainloader))
  print(len(testloader))
  print(len(testloaderhard))
  time_epoch_start = time.time()
  
  for batch_idx, (real_sample, cond,tokenized_mol) in enumerate(trainloader): # real sample are the real images and cond are the molecules (for of list len(bs))
      print(batch_idx)
      #print(real_sample.shape)
      #print(len(cond))
      print(len(cond))
      print(tokenized_mol.shape)

      
      break
      
      '''
      # to save images from the current batch in one channel per one channel
      for ch in range(5):
          ch_img = real_sample[:,ch:ch+1,:,:]
          save_image(ch_img, os.path.join(args.save_dir, 'examples%s_%s_real.png' % (batch_idx, ch)), 
                      normalize=True, nrow=10, range=(-0.5, 0.5))

      
      # to save images from the current batch in three channel visu
      ch_img = get_3c_image(real_sample)
      save_image(ch_img, os.path.join(args.save_dir, 'examples%s_%s_real_3channel.png' % (batch_idx, ch)), 
                  normalize=True, nrow=10, range=(-0.5, 0.5))

      break
      '''
  time_epoch_end=time.time()
  time_epoch=time_epoch_end-time_epoch_start
  print('time for one epoch: ',time_epoch)
  print('approx time for one batch: ',time_epoch/len(trainloader))
  