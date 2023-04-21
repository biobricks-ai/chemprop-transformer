import torch
from CVAE.model import CVAE
from utils import loadTest, prepareInputs, embeddingAccuracy
from tqdm import tqdm
import numpy as np
from rdkit import Chem, DataStructs

import json
import dvc.api
import os

def test(model, vocab, metricsOutPath, processedData):
    model.eval()
    
    testTensors = loadTest(processedData)
    testTensors = testTensors[:1000]
    
    totalAccuracy = 0
    validGenerations = 0
    avgTanimoto = 0
    
    for data in tqdm(testTensors):
        embedding, labels = prepareInputs(data, model.vocabSize, device)
        
        xHat, _, _, _ = model(embedding, labels)
        
        indexedOri = list(data[0])
        indexedPred = torch.argmax(xHat, dim=1).tolist()
        
        oriSmiles = ''.join([vocab[i] for i in indexedOri]).replace(' ', '')
        genSmiles = ''.join([vocab[i] for i in indexedPred]).replace(' ', '')
        
        mol = Chem.MolFromSmiles(genSmiles)
        if mol is not None:
            validGenerations += 1
            oriMol = Chem.MolFromSmiles(oriSmiles)
            genFp = Chem.RDKFingerprint(mol)
            goriFp = Chem.RDKFingerprint(oriMol)
            Tan = DataStructs.TanimotoSimilarity(genFp,goriFp)
            avgTanimoto += Tan
            
        embedAcc = embeddingAccuracy(indexedOri, indexedPred)
        totalAccuracy += embedAcc
    
    if validGenerations == 0:
        avgTanimoto = 0
    else:
        avgTanimoto = avgTanimoto/validGenerations
    
    with open('{}accuracy.tsv'.format(metricsOutPath), 'w') as f:
        f.write('generated\tvalid\tavg tanimoto\tratio\tgeneral accuracy\n')
        f.write('{}\t{}\t{}\t{}\t{}\n'.format(len(testTensors), validGenerations, avgTanimoto, validGenerations/len(testTensors), totalAccuracy/len(testTensors)))

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    params = dvc.api.params_show()
    
    latentDim = params['latentDim']
    
    modelFolder = params['testing']['modelFolder']
    processedData = params['testing']['processedData']
    metricsOutPath = params['testing']['metricsOutPath']
    
    os.makedirs(metricsOutPath, exist_ok=True)
    
    with open('{}/modelInfo.json'.format(processedData)) as f:
        modelInfo = json.load(f)
    
    with open('{}/vocabs.json'.format(processedData)) as f:
        vocab = json.load(f)['smileVocab']
    
    embeddingSize = modelInfo['embeddingSize']
    vocabSize = modelInfo['vocabSize']
    assaySize = modelInfo['assaySize']
    
    labelsSize = assaySize + 1
    
    model = CVAE(embeddingSize, vocabSize, labelsSize, latentDim)
    model = model.to(device)
    
    model.load_state_dict(torch.load('{}bestModel.pt'.format(modelFolder)))
    
    test(model, vocab, metricsOutPath, processedData)
    