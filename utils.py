import torch
from torch import optim
import logging
import os
import json
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
                
def save_img_as_npz(img, fname):
    img = img.numpy()
    img = img.transpose(1,2,0)
    img = img + 0.5 # undo normalization
    img_dict = {}
    img_dict['sample'] = img

    np.savez(fname, **img_dict)



def create_dir(dirname):
    try:
        os.mkdir(dirname)
        return True
    except OSError:
        return False


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

def get_ecfp_tensor(mols):
    ecfp=get_ecfp6_fingerprints(mols)
    ecfp_tensor=torch.from_numpy(ecfp)
    ecfp_tensor=ecfp_tensor.float()
    return ecfp_tensor