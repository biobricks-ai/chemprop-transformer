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

smiCharSet, smiCharToInt, _ = generateCharSet(activities['smiles'], maxEmbeddingSize)

activities['smiles'] = activities['smiles'].progress_apply(lambda smiles: smiles+smiCharSet[-1])
activities = activities[activities['smiles'].str.len() <= maxEmbeddingSize]

activities['Occur'] = activities.groupby('assay')['value'].transform('size')
activities = activities[activities['Occur'] >= 500]

#--------------------------------------------------------------------------
# reduced dataset for tsne testing

a1 = activities[activities['assay'] == '8bd16db6-124f-447c-98c1-d2a86849b333']
a2 = activities[activities['assay'] == 'e597a518-81c2-41bf-b876-0e09f961ceb6']

a1 = a1.sample(n=10000)
a2 = a2.sample(n=10000)

activities = pd.concat([a1, a2])

# activities = activities.sample(n=5000)

uniqueAssays = activities['assay'].unique().tolist()
maxAssaySize = len(uniqueAssays)
assayIndexMap = {assay:i for i,assay in enumerate(uniqueAssays)}

#--------------------------------------------------------------------------
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
trainTensors = np.array(trainTensors, dtype=object)

testTensors = testSet.values.tolist()
testTensors = np.array(testTensors, dtype=object)

validTensors = validSet.values.tolist()
validTensors = np.array(validTensors, dtype=object)

np.save('{}/train.npy'.format(outDataFolder), trainTensors)
np.save('{}/test.npy'.format(outDataFolder), testTensors)
np.save('{}/valid.npy'.format(outDataFolder), validTensors)