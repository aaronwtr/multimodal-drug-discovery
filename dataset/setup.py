from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from .cellpainting import *

import numpy as np

def my_collate(batch):
    ''' Custom collate function that filters out empty samples '''
    batch = list(filter (lambda x:x is not None, batch))
    return default_collate(batch)

def data_sampler(dataset, batch_size, num_workers, shuffle=True):
    ''' Helper for sampling data from full dataset '''
    loader = DataLoader(dataset, shuffle=shuffle, drop_last=True, batch_size=batch_size, num_workers=num_workers, collate_fn=my_collate)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, collate_fn=my_collate)
            loader = iter(loader)
            yield next(loader)

def setup_dataloaders(args):
    if args.dataset == 'cell-painting':
        print('loading normal dataset')
        trainset = CellPaintingDataset(datadir=args.datadir, metafile=args.train_metafile, img_size=args.img_size, featfile=args.featfile)
        valset = CellPaintingDataset(datadir=args.datadir, metafile=args.val_metafile, img_size=args.img_size, featfile=args.featfile)
        valset_hard=CellPaintingDataset(datadir=args.datadir, metafile=args.val_hard_metafile, img_size=args.img_size)
    elif args.dataset == 'cell-painting-patch':
        assert(args.img_size > 64)
        print('loading patch dataset')
        trainset = CellPaintingPatchDataset(datadir=args.datadir, metafile=args.train_metafile, img_size=args.img_size)
        valset = CellPaintingPatchDataset(datadir=args.datadir, metafile=args.val_metafile, img_size=args.img_size)
        valset_hard=CellPaintingPatchDataset(datadir=args.datadir, metafile=args.val_hard_metafile, img_size=args.img_size)
    
    elif args.dataset == 'cell-painting-smiles':
        assert(args.img_size > 64)
        print('loading transformers dataset')
        trainset = CellPaintingDataset_SMILES(datadir=args.datadir, metafile=args.train_metafile,tokenizer=args.tokenizer, img_size=args.img_size, featfile=args.featfile)
        valset = CellPaintingDataset_SMILES(datadir=args.datadir, metafile=args.val_metafile,tokenizer=args.tokenizer, img_size=args.img_size, featfile=args.featfile)
        valset_hard=CellPaintingDataset_SMILES(datadir=args.datadir, metafile=args.val_hard_metafile,tokenizer=args.tokenizer, img_size=args.img_size, featfile=args.featfile)
    
    else:
        raise KeyError('Dataset %s is not valid' % args.dataset)

    
    #trainloader = iter(data_sampler(dataset=trainset, batch_size=args.batch_size, num_workers=args.num_workers))

    trainloader=DataLoader(dataset=trainset, batch_size=args.batch_size_train, num_workers=args.num_workers)

    if valset is not None:
        valloader = DataLoader(dataset=valset, batch_size=args.batch_size_val, num_workers=args.num_workers, 
                               drop_last=False, shuffle=args.use_nce_loss)
        valloader_hard = DataLoader(dataset=valset_hard, batch_size=args.batch_size_val_hard, num_workers=args.num_workers, 
                               drop_last=False, shuffle=args.use_nce_loss)
    else:
        valloader = None

    return trainloader, valloader, valloader_hard
