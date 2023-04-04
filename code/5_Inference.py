import torch
from CVAE.model import CVAE
from utils import loadTest, idTo1Hot, embeddingAccuracy
from tqdm import tqdm
import numpy as np
from rdkit import Chem, DataStructs

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import json
import dvc.api
import os


def inference(model):
    model = model.encoder
    model.eval()
    
    testTensors = loadTest('data/processed/')
    
    for data in tqdm(testTensors):
        embeddings = np.array([idTo1Hot(i, 58) for i in list(data[0])])
        embeddings = torch.Tensor(embeddings)
        embeddings = embeddings.to(device)
        embeddings = embeddings.view(-1)
        
        sequence = data[1]
        sequence = torch.Tensor(sequence)
        sequence = sequence.to(device)
        
        interaction = data[2]
        interaction = torch.Tensor(interaction)
        interaction = interaction.to(device)
        
        labels = torch.cat((sequence, interaction), dim=0)
        
        zMean, zLogvar = model(embeddings, labels)
        
        std = torch.exp(0.5*zLogvar)
        eps = torch.randn_like(std)
        
        z = zMean + eps*std
        
        plt.scatter(z.tolist(), z.tolist()[::-1])
        plt.savefig('scatterResults.png')
        
        



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    params = dvc.api.params_show()
    
    embeddingDim = params['embeddingDim']
    labelsDim = params['labelsDim']
    latentDim = params['latentDim']
    
    modelPath = params['demo']['modelPath']
    
    model = CVAE(embeddingDim, labelsDim, latentDim).to(device)
    model.load_state_dict(torch.load(modelPath))
    
    inference(model)