import torch
from CVAE.model import CVAE
from utils import loadTest, idTo1Hot, embeddingAccuracy
from tqdm import tqdm
import numpy as np
from rdkit import Chem, DataStructs

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
        
        print(z)



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    params = dvc.api.params_show()
    
    embeddingDim = params['embeddingDim']
    labelsDim = params['labelsDim']
    latentDim = params['latentDim']
    
    vocabPath = params['testing']['vocabPath']
    modelPath = params['testing']['modelPath']
    metricsOutPath = params['testing']['metricsOutPath']
    
    model = CVAE(embeddingDim, labelsDim, latentDim).to(device)
    model.load_state_dict(torch.load(modelPath))
    
    inference(model)