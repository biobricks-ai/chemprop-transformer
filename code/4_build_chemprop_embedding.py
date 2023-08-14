from __future__ import print_function
from tqdm import tqdm

import json, os, signal, dvc.api, numpy as np, pathlib, time
import torch, torch.nn.functional as F, torch.optim as optim
import torch.nn as nn

import torch.utils.data
from torch.utils.data import Dataset, Subset

import cvae.models
import cvae.utils as utils
import cvae.models.triplet  as triplet
import cvae.models.predvae

# load the model
device = torch.device(f'cuda:0')
# pvae = cvae.models.triplet.IsometricEmbedding.load("brick/isoembed2.pt").to(device)
pvae = cvae.models.predvae.PredVAE.load("brick/pvae.pt").to(device)

def build_chemprop_embedding(infile, outfile, batch_size=1024):
    
    # insmi, inlbl, inval, morgan, invae, trnflag
    T = torch.load(infile).tensors
    ismiles, imorgan, ilbl, ival = (0,3,1,2)
    insmi, inchem, inlbl, inval = T[ismiles], T[imorgan], T[ilbl], T[ival]

    # Create a DataLoader for the input data
    input_dataset = torch.utils.data.TensorDataset(insmi, inchem, inlbl, inval)
    input_dataloader = torch.utils.data.DataLoader(input_dataset, batch_size=batch_size)

    # Initialize an empty list to store the results
    # vaeresults = []

    # # Process data in batches
    # for batch_inchem, batch_inlbl, batch_inval in tqdm(input_dataloader):
        
    #     batch_inchem, batch_inlbl = batch_inchem.to(device), batch_inlbl.to(device)
    #     _, z_mean, z_logvar = mvae(batch_inchem.to(torch.float), batch_inlbl)

    #     # Detach from the computation graph
    #     z_mean = z_mean.detach()
    #     z_logvar = z_logvar.detach()

    #     # Concatenate results and append to the results list
    #     batchvae = torch.cat((z_mean, z_logvar), dim=1)
    #     vaeresults.append(batchvae)
    
    pvaeresults = []
    for ten in tqdm(input_dataloader):
        bsmi, bchem, blbl, bval = [t.to(device) for t in ten]
        decsmi, zmean, zlogvar = pvae(bsmi)
        pvaeresults.append(zmean.detach())

    # Concatenate the results and create a new tensor
    # tmvae = torch.cat(vaeresults, dim=0)
    tpvae = torch.cat(pvaeresults, dim=0)
    
    # Append the new tensor to the original tensors list
    T = T + (tpvae,)

    # Create a new dataset with the new tensors
    td = torch.utils.data.TensorDataset(*T)

    # Save the dataset to a file
    torch.save(td, outfile)

# save train tensordataset with predvae embeddings
infile = utils.res("data/processed/train.pt")
outfile = utils.res("data/biosim/train.pt")
build_chemprop_embedding(infile, outfile)

# save validation tensordataset with predvae embeddings
infile = utils.res("data/processed/validation.pt")
outfile = utils.res("data/biosim/validation.pt")
build_chemprop_embedding(infile, outfile)