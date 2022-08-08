import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize
from .utils import ResizeTensor, CropPatch

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

import os

from albumentations import (
    Resize, 
    CenterCrop, 
    Normalize,   
    RandomCrop,
    Compose,
    Transpose,
    HorizontalFlip,
    VerticalFlip,    
    Compose,
    Transpose,
    RandomRotate90,
)

from albumentations.pytorch.transforms import ToTensorV2


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
    ecfp_tensor=torch.from_numpy(fps)
    ecfp_tensor=ecfp_tensor.float()
    return(ecfp_tensor)





class CustomTransform(object):
    def __init__(self, mode, img_size=512, original_size=512):
        if mode == 'train':
            transforms=Compose([RandomCrop(original_size,original_size),VerticalFlip(p=0.5),HorizontalFlip(p=0.5),Normalize((0.5, 0.5, 0.5, 0.5, 0.5), (1., 1., 1., 1., 1.)),ToTensorV2()])
        elif mode == 'val' or mode == 'test':
            transforms = [CenterCrop(original_size,original_size),Normalize((0.5, 0.5, 0.5, 0.5, 0.5), (1., 1., 1., 1., 1.)),ToTensorV2()]  
        else:
            raise KeyError("mode %s is not valid, must be 'train' or 'val' or 'test'" % mode)

        self.transforms = transforms
        self.to_tensor = ToTensorV2()
        self.normalize = Normalize((0.5, 0.5, 0.5, 0.5, 0.5), (1., 1., 1., 1., 1.))
        self.resize = ResizeTensor(image_size=img_size, original_size=original_size)
    
    def __call__(self, imgs):
        augmented = self.transforms(image=imgs)
        imgs=augmented['image'] 
        imgs = self.resize(imgs)
        return imgs



class CellPaintingDataset(Dataset):
    ''' Base Dataset class '''

    def __init__(self, datadir, metafile, mode="train", img_size=512, featfile=None):
        self.datadir = datadir
        self.metadata = pd.read_csv(metafile)
        self.molfeats = pd.read_csv(featfile, index_col=1) if featfile is not None else None
        self.transforms = CustomTransform(mode=mode, img_size=img_size)

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        try:
            sample = self.get_sample_img(idx)
            sample.update(self.get_sample_mol(idx))

        except Exception as e:
            print(e)
            return None
        
        return sample['image'], sample['feat']

    def load_img(self, key):
        ''' Load image from key '''
        img = np.load(os.path.join(self.datadir, "%s.npz" % key))
        img = img["sample"] # Shape 520 x 696 x 5
        #img = [img[:,:,j] for j in range(5)]
        img = self.transforms(img)

        return img
    
    def get_sample_img(self, idx):
        '''Returns a dict corresponding to sample img for the provided index'''
        sample = self.metadata.iloc[idx]
        key = sample['SAMPLE_KEY']

        # load 5-channel image
        img = self.load_img(key)

        return {'key_img': key, 'image': img}

    def get_sample_mol(self, idx):
        ''' Returns a dict corresponding to sample molecule for the provided index'''
        sample = self.metadata.iloc[idx]
        smiles = sample['SMILES']

        if self.molfeats is not None:
            feat = self.molfeats.loc[smiles]['FEAT']
            feat = eval(f"np.array({feat})")
            feat = torch.from_numpy(feat).float()
            return {'key_chem': sample['SAMPLE_KEY'], 'feat': feat}

        return {'key_chem': sample['SAMPLE_KEY'], 'feat': smiles}

class CustomTransformPatch(CustomTransform):
    def __init__(self, mode, img_size=512, original_size=512, patch_size=64):
        super().__init__(mode, img_size=img_size, original_size=original_size)
        if mode == 'train':
            self.patch_crop = CropPatch(patch_size=patch_size, random=True)
        else:
            self.patch_crop = CropPatch(patch_size=patch_size, random=False)

    def __call__(self, imgs):
        imgs = super().__call__(imgs)
        imgs = self.patch_crop(imgs)
        return imgs

class CellPaintingPatchDataset(CellPaintingDataset):
    def __init__(self, datadir, metafile, mode="train", img_size=512):
        super().__init__(datadir=datadir, metafile=metafile, mode=mode, img_size=img_size)
        self.transforms = CustomTransformPatch(mode=mode, img_size=img_size)






# modified cell painting dataset for transformer use adapted from romain (meaning adding the tokenization as dataloder output)
class CellPaintingDataset_SMILES(Dataset):
    ''' Base Dataset class '''

    def __init__(self,
                datadir, 
                metafile,
                encoding_text="SMILES",
                text_len=256,
                truncate_captions=True,
                tokenizer=None,
                mode="train",
                img_size=512,
                featfile=None):

        self.datadir = datadir
        self.metadata = pd.read_csv(metafile)
        self.molfeats = pd.read_csv(featfile, index_col=1) if featfile is not None else None
        self.transforms = CustomTransform(mode=mode, img_size=img_size)

        self.encoding_text = encoding_text
        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        try:
            sample = self.get_sample_img(idx)
            sample.update(self.get_sample_mol(idx))
            sample.update(self.get_tokenized_mol(idx))

        except Exception as e:
            print(e)
            return None
        
        return sample['image'], sample['feat'],sample['tokenized_mol']

    def load_img(self, key):
        ''' Load image from key '''
        img = np.load(os.path.join(self.datadir, "%s.npz" % key))
        img = img["sample"] # Shape 520 x 696 x 5
        img = self.transforms(img)

        return img
    
    def get_sample_img(self, idx):
        '''Returns a dict corresponding to sample img for the provided index'''
        sample = self.metadata.iloc[idx]
        key = sample['SAMPLE_KEY']

        # load 5-channel image
        img = self.load_img(key)

        return {'key_img': key, 'image': img}

    def get_sample_mol(self, idx):
        ''' Returns a dict corresponding to sample molecule for the provided index'''
        sample = self.metadata.iloc[idx]
        smiles = sample['SMILES']

        if self.molfeats is not None:
            feat = self.molfeats.loc[smiles]['FEAT']
            feat = eval(f"np.array({feat})")
            feat = torch.from_numpy(feat).float()
            return {'key_chem': sample['SAMPLE_KEY'], 'feat': feat}

        return {'key_chem': sample['SAMPLE_KEY'], 'feat': smiles}
    
    def get_tokenized_mol(self,idx):
        sample = self.metadata.iloc[idx]
        #smiles=self.get_sample_mol(idx)['feat']
        text = self.metadata.loc[idx, 'SMILES']

        text_tokenized = self.tokenizer.tokenize(
                text,
                self.text_len,
                truncate_text=self.truncate_captions
            ).squeeze(0)
        
        return {'key_chem': sample['SAMPLE_KEY'], 'tokenized_mol': text_tokenized}