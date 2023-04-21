from __future__ import print_function
import torch

torch.set_default_tensor_type(torch.FloatTensor)

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

class Encoder(nn.Module):
    def __init__(self, embeddingSize, labelsSize, latentSize):
        super(Encoder, self).__init__()
        self.embeddingSize = embeddingSize
        self.labelsSize = labelsSize
        self.latentSize = latentSize
        
        self.kernelSize = 9
        self.strideSize = 2
        
        zSize = embeddingSize + labelsSize
        
        for _ in range(3): # for num of conv layers
            zSize = ((zSize - self.kernelSize)//self.strideSize) + 1
            
        self.relu = nn.ReLU()
        self.conv0 = nn.Conv1d(1, 16, kernel_size = self.kernelSize, stride = self.strideSize)
        self.conv1 = nn.Conv1d(16, 16, kernel_size = self.kernelSize, stride = self.strideSize)
        self.conv2 = nn.Conv1d(16, 1, kernel_size = self.kernelSize, stride = self.strideSize)
        
        self.zMean = nn.Linear(zSize, latentSize)
        self.zLogVar = nn.Linear(zSize, latentSize)
        
        
    def forward(self, smiEmbedding, labelsEmbedding):
        catEmbeddings = torch.cat((smiEmbedding, labelsEmbedding), dim=0)
        catEmbeddings = catEmbeddings.view(1, 1, catEmbeddings.size(0))
        z = self.conv0(catEmbeddings)
        z = self.relu(z)
        z = self.conv1(z)
        z = self.relu(z)
        z = self.conv2(z)
        z = self.relu(z)
        z = z.view(-1)
        
        mean = self.zMean(z)
        # mean = F.softmax(mean, 0)
        logVar = self.zLogVar(z)
        # logVar = F.softmax(logVar, 0)
        return mean, logVar

    
class Decoder(nn.Module):
    def __init__(self, encodeSize, labelsSize, outputSize, nGru = 3, hiddenSize = 4096):
        super(Decoder, self).__init__()
        
        self.encodeSize = encodeSize
        self.labelsSize = labelsSize
        self.outputSize = outputSize
        self.hiddenSize = hiddenSize
        self.nGru = nGru
        
        self.gruInputSize = self.encodeSize + self.labelsSize
        
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.norm1 = nn.BatchNorm1d(1)
        
        self.gru = nn.GRU(self.gruInputSize, self.hiddenSize, self.nGru, batch_first = True)
        self.l2 = nn.Linear(self.hiddenSize, self.outputSize)

    def forward(self, encodedInput, labels):
        catEmbeddings = torch.cat((encodedInput, labels), dim=0)
        catEmbeddings = catEmbeddings.view(1, 1, catEmbeddings.size(0))
        yPred, yHidden = self.gru(catEmbeddings)
        yPred = self.norm1(yPred)
        yPred = self.dropout(yPred)
        yPred =  self.l2(yPred)
        yPred = self.relu(yPred)
        return yPred
  
  
class CVAE(nn.Module):
    def __init__(self, embeddingSize, vocabSize, labelsSize, latentSize):
        super(CVAE, self).__init__()
        
        self.embeddingSize = embeddingSize
        self.vocabSize = vocabSize
        self.labelsSize = labelsSize
        self.latentSize = latentSize
        
        self.oneHotEmbeddingSize = embeddingSize * vocabSize
        
        self.dropout = nn.Dropout(p=0.1)
        
        self.encoder = Encoder(self.oneHotEmbeddingSize, self.labelsSize, self.latentSize)
        self.decoder = Decoder(self.latentSize, self.labelsSize, self.oneHotEmbeddingSize)
        
    def reparametrize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps*std
        
        
    def forward(self, embedding, labels):
        embedding = embedding.view(-1)
        zMean, zLogvar = self.encoder(embedding, labels)
        latentSpace = self.reparametrize(zMean, zLogvar)
        z = self.dropout(latentSpace)
        yPred = self.decoder(z, labels)
        yPred = yPred.view(self.embeddingSize, self.vocabSize)
        return yPred, zMean, zLogvar, latentSpace.tolist()