from __future__ import print_function
from tqdm import tqdm

import json, os, signal, dvc.api, numpy as np, pathlib, time
import torch, torch.nn.functional as F, torch.optim as optim, torch.utils.data
import torch.nn as nn


class BioSim(nn.Module):

    def __init__(self, vaedim, numlbl):
        super(BioSim, self).__init__()
        
        self.lbl_embed = nn.Sequential(nn.Embedding(numlbl, vaedim),nn.ReLU())
        self.chemlbl = nn.Sequential(
            nn.Linear(vaedim*2, vaedim),nn.ReLU(), 
            nn.Linear(vaedim, vaedim),nn.ReLU())
        
    def forward(self, invae, inlbl):
        eml = self.lbl_embed(inlbl)
        cat = torch.cat([invae, eml], dim=1)
        chemlbl = self.chemlbl(cat)
        dot = torch.sum(invae*chemlbl,dim=1)
        return dot
    
    @staticmethod
    def loss(pred, inval):
        return F.binary_cross_entropy_with_logits(pred, inval)
    
    @staticmethod
    def load(ptfile):
        state_dict = torch.load(ptfile)
        
        # Extract vaedim and numlbl from the state_dict
        vaedim = state_dict['chemlbl.0.weight'].size(1) // 2
        numlbl = state_dict['lbl_embed.0.weight'].size(0)

        # Create a new instance of the BioSim class with the appropriate dimensions
        model = BioSim(vaedim, numlbl)

        # Load the saved state_dict into the new model instance
        model.load_state_dict(state_dict)

        return model