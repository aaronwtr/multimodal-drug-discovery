# Multimodal-drug-discovery

Project by OpenBioML community to learn multi modal representation learning from structure and other modalities (images,text, ..).


## Image-structure dataset

A dataset of 30K compound and ~1M images is available here https://github.com/gigascience/paper-bray2017.
Quite messy to dowload everything with the metadata, also need to remove DMSO (no compound), they did those kind of things in https://openreview.net/pdf?id=OdXKRtg1OG
but did not release their code or dataset yet (will be updated on https://github.com/ml-jku/cloome).

There is a subset of this dataset containing 10.5K compound with their associated images and could be a way to start since it is quite easy to download and the metadata
are also easily available at https://github.com/ml-jku/hti-cnn with the metadata in datasplit. 

## Get started with few data

We will start by a subset of this to experiment quickly ( 1/10 of the data)
To download this dataset go to https://ml.jku.at/software/cellpainting/dataset/ (in https://github.com/ml-jku/hti-cnn) and download dataset00 then untar in some folder like images00. The metadata associated are in data/metadata. There is one train metadata and two test sets. Test_easy is molecules that the network as seen during training but not their related images and test_hard is molecules than the network has never seen.
Some dataloader functions that returns image/structure pairs are in dataset. There is an example on how it works on test_dataloader.py 

## Get started with baseline

The file clip_test.py is a first simple baseline for clip framework to start from a baseline

