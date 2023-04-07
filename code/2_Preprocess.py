import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


tqdm.pandas()

import dvc.api
from utils import generateCharSet, toEmbedding, idTo1Hot
import os

params = dvc.api.params_show()["preprocessing"] 
rawDataPath = params["rawDataPath"]
outDataFolder = params["outDataFolder"]

os.makedirs(outDataFolder, exist_ok=True)

data = pd.read_csv(rawDataPath)[['smiles','assay','value']]

uniqueAssays = data['assay']
uniqueAssays = set(uniqueAssays)
uniqueAssays = list(uniqueAssays)

maxEmbeddingSize = 244 #data['smiles'].str.len().max()
maxAssaySize = len(uniqueAssays)

assayIndexMap = {assay:i for i,assay in enumerate(uniqueAssays)}

smiCharSet, smiCharToInt, _ = generateCharSet(data['smiles'], maxEmbeddingSize)

data['smiles'] = data['smiles'].progress_apply(lambda smiles: toEmbedding(smiles, smiCharToInt, maxEmbeddingSize))
data['assay'] = data['assay'].progress_apply(lambda assay: idTo1Hot(assayIndexMap[assay], maxAssaySize))
data['value'] = data['value'].progress_apply(lambda value: torch.tensor([value]))


modelInfo = {
    'embeddingSize' : maxEmbeddingSize,
    'vocabSize': len(smiCharSet),
    'assaySize': maxAssaySize,
}

with open('{}/modelInfo.json'.format(outDataFolder), 'w') as f:
    json.dump(modelInfo, f, indent=4)

vocabs = {
    'smileVocab': smiCharSet,
    'assayMap': assayIndexMap
}

with open('{}/vocabs.json'.format(outDataFolder), 'w') as f:
    json.dump(vocabs, f, indent=4)


trainSet = data.sample(frac=.7)
data = data.drop(trainSet.index)
testSet = data.sample(frac=.5)
validSet = data.drop(testSet.index)

trainTensors = trainSet.values.tolist()
testTensors = testSet.values.tolist()
validTensors = validSet.values.tolist()

np.save('{}/train.npy'.format(outDataFolder), trainTensors)
np.save('{}/test.npy'.format(outDataFolder), testTensors)
np.save('{}/valid.npy'.format(outDataFolder), validTensors)