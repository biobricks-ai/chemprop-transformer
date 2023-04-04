from __future__ import print_function
import torch

torch.set_default_tensor_type(torch.FloatTensor)

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

class Encoder(nn.Module):
    def __init__(self, embeddingDim, labelsDim, latentDim):
        super(Encoder, self).__init__()
        self.embeddingDim = embeddingDim
        self.labelsDim = labelsDim
        self.latentDim = latentDim
        
        self.kernelSize = 9
        self.strideSize = 2

        self.relu = nn.ReLU()
        self.conv0 = nn.Conv1d(1, 16, kernel_size = self.kernelSize, stride = self.strideSize)
        self.conv1 = nn.Conv1d(16, 16, kernel_size = self.kernelSize, stride = self.strideSize)
        self.conv2 = nn.Conv1d(16, 1, kernel_size = self.kernelSize, stride = self.strideSize)
        
        zSize = embeddingDim + labelsDim
        
        for _ in range(3): # for num of conv layers
            zSize = ((zSize - self.kernelSize)//self.strideSize) + 1
        
        self.zMean = nn.Linear(zSize, latentDim)
        self.zLogVar = nn.Linear(zSize, latentDim)

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
        mean = F.softmax(mean, 0)
        logVar = self.zLogVar(z)
        logVar = F.softmax(logVar, 0)
        return mean, logVar
    
class Decoder(nn.Module):
    def __init__(self, embeddingDim, labelsDim, latentDim):
        super(Decoder, self).__init__()
        self.embeddingDim = embeddingDim
        self.labelsDim = labelsDim
        self.latentDim = latentDim

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        
        self.gru1 = nn.GRU(latentDim + labelsDim, 4096, 3, batch_first = True)
        self.norm1 = nn.BatchNorm1d(1)
        self.l2 = nn.Linear(4096, embeddingDim)

    def forward(self, encodedEmbedding, labelsEmbedding):
        catEmbeddings = torch.cat((encodedEmbedding, labelsEmbedding), dim=0)
        catEmbeddings = catEmbeddings.view(1, 1, catEmbeddings.size(0))
        yPred, yHidden = self.gru1(catEmbeddings)
        yPred = self.norm1(yPred)
        yPred = self.dropout(yPred)
        yPred =  self.l2(yPred)
        yPred = self.relu(yPred)
        return yPred
    
class CVAE(nn.Module):
    #244 12 64
    def __init__(self, embeddingDim, labelsDim, latentDim):
        super(CVAE, self).__init__()
        self.embeddingDim = embeddingDim
        self.labelsDim = labelsDim
        self.latentDim = latentDim

        self.encoder = Encoder(embeddingDim, labelsDim, latentDim)
        self.dropout = nn.Dropout(p=0.1)
        self.decoder = Decoder(embeddingDim, labelsDim, latentDim)

    def reparametrize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps*std

    def forward(self, embedding, labels):
        embedding = embedding.view(-1)
        zMean, zLogvar = self.encoder(embedding, labels)
        z = self.reparametrize(zMean, zLogvar)
        z = self.dropout(z)
        yPred = self.decoder(z, labels)
        yPred = yPred.view(244, 58)
        return yPred, zMean, zLogvar