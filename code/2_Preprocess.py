import json
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
tqdm.pandas()

import dvc.api

from utils import generateCharSet, toEmbedding, idTo1Hot
import os

params = dvc.api.params_show()["preprocessing"] 
rawDataPath = params["rawDataPath"]
outDataFolder = params["outDataFolder"]

os.makedirs(outDataFolder, exist_ok=True)

maxEmbeddingSize = params["maxEmbeddingSize"]
maxEmbeddingSize = 60 #data['smiles'].str.len().max()

activities = pd.read_csv(rawDataPath)[['smiles','assay','value']]
activities = activities[activities['smiles'].str.len() <= maxEmbeddingSize]

uniqueAssays = activities['assay'].unique().tolist()
maxAssaySize = len(uniqueAssays)
assayIndexMap = {assay:i for i,assay in enumerate(uniqueAssays)}

smiCharSet, smiCharToInt, _ = generateCharSet(activities['smiles'], maxEmbeddingSize)

activities['smiles'] = activities['smiles'].progress_apply(lambda smiles: toEmbedding(smiles, smiCharToInt, maxEmbeddingSize))
activities['assay'] = activities['assay'].progress_apply(lambda assay: idTo1Hot(assayIndexMap[assay], maxAssaySize))
activities['value'] = activities['value'].progress_apply(lambda value: torch.tensor([value]))

modelInfo = {
    'embeddingSize' : maxEmbeddingSize,
    'vocabSize': len(smiCharSet),
    'assaySize': maxAssaySize,
}

with open('{}/modelInfo.json'.format(outDataFolder), 'w') as f:
    json.dump(modelInfo, f, indent=4)

vocabs = {
    'smileVocab': smiCharSet,
    'assayMap': uniqueAssays
}

with open('{}/vocabs.json'.format(outDataFolder), 'w') as f:
    json.dump(vocabs, f, indent=4)


trainSet = activities.sample(frac=.7)
activities = activities.drop(trainSet.index)
testSet = activities.sample(frac=.5)
validSet = activities.drop(testSet.index)

trainTensors = trainSet.values.tolist()
testTensors = testSet.values.tolist()
validTensors = validSet.values.tolist()

np.save('{}/train.npy'.format(outDataFolder), trainTensors)
np.save('{}/test.npy'.format(outDataFolder), testTensors)
np.save('{}/valid.npy'.format(outDataFolder), validTensors)