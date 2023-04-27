from __future__ import print_function
import torch, torch.nn as nn, torch.nn.functional as F, torch.utils.data
torch.set_default_tensor_type(torch.FloatTensor)

class Encoder(nn.Module):
    
    def __init__(self, smiles_padlength, smiles_vocabsize, assays_vocabsize, latentdim, lbldim):
        """
        smiles_padlength: length of the padded smiles
        smiles_vocabsize: size of the smiles vocabulary
        assays_vocabsize: size of the assays vocabulary
        latentdim: dimension of the encoded latent space
        """
        super(Encoder, self).__init__()
        
        smidim = 20
        
        kernel = 9
        stride = 1
        convdim = 10
        
        zsize = smiles_padlength
        for _ in range(3): # for num of conv layers
            zsize = ((zsize - kernel)//stride) + 1
        zsize = zsize + lbldim + 1
        
        self.lbl_embed = nn.Embedding(assays_vocabsize, lbldim)
        self.smi_embed = nn.Embedding(smiles_vocabsize, smidim)
        
        self.arelu = nn.ReLU()
        self.conv1 = nn.Conv1d(smidim, convdim, kernel_size = kernel, stride = stride)
        self.conv2 = nn.Conv1d(convdim, convdim, kernel_size = kernel, stride = stride) # 4, 1, 240
        self.conv3 = nn.Conv1d(convdim, 1, kernel_size = kernel, stride = stride)
        
        self.hidden = nn.Linear(zsize, latentdim)
        self.zMean = nn.Linear(latentdim, latentdim) # smi, lbl, val are concatenated before this
        self.zlvar = nn.Linear(latentdim, latentdim)
        
        
    def forward(self, smilesinput, labelsinput, valueinput):
        # smilesinput, labelsinput, valueinput = trn[1:5][0], trn[1:5][1], trn[1:5][2]
        lblEmbedding = self.lbl_embed(labelsinput).to(torch.float) # 4, 10
        valinput2 = valueinput.unsqueeze(1).to(torch.float)
        
        z1 = self.smi_embed(smilesinput).permute(0,2,1) # 4, 10, 244 
        z2 = self.arelu(self.conv1(z1)) # 4, 10, 242
        z3 = self.arelu(self.conv2(z2)) # 4, 1, 240
        z4 = self.arelu(self.conv3(z3)).squeeze(1) # 4, 238
        z5 = self.arelu(self.hidden(z4))
        z6 = torch.cat((z5, lblEmbedding, valinput2), dim=1)
        
        mean = self.zMean(z6)
        logvar = self.zlvar(z6)
        return mean, logvar, lblEmbedding

    
class Decoder(nn.Module):
    
    def __init__(self, smiles_padlength, smiles_vocabsize, assays_vocabsize,
                 sampledim, lbldim=10, nGru = 2, 
                 hiddensize = 512):
        """
        sampledim: dimension of the sample vector
        assays_vocabsize: size of the assays vocabulary
        lbldim: dimension of the label embedding
        smiles_padlength: length of the padded smiles
        nGru: number of GRU layers
        hiddensize: hidden size of the GRU layers
        """
        super(Decoder, self).__init__()
        
        self.padsize = smiles_padlength
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(sampledim+lbldim+1,128)
        self.gru = nn.GRU(128, smiles_vocabsize, nGru, batch_first = True)
        # self.l2 = nn.Linear(hiddensize, smiles_vocabsize)
    

    def forward(self, encodedinput, embedlbl, valueinput):
        valinput = valueinput.unsqueeze(1).to(torch.float)
        embedall = torch.cat((encodedinput, embedlbl, valinput), dim=1)
        embedall = self.relu(self.l1(embedall)) # 4, 523
        embedall = embedall.unsqueeze(1) # 4, 1, 523
        embedrep = embedall.repeat(1,self.padsize,1) # 4, 244, 523
        out, _ = self.gru(embedrep) # 4, 244, hiddensize
        
        return out 
  
  
class CVAE(nn.Module):
    def __init__(self, smiles_padlength, smiles_vocabsize, assays_vocabsize, latentdim):
        
        super(CVAE, self).__init__()
        
        self.dropout = nn.Dropout(p=0.1)
        
        lbldim = 5
        self.encoder = Encoder(smiles_padlength, smiles_vocabsize, assays_vocabsize, latentdim, lbldim)
        self.decoder = Decoder(smiles_padlength, smiles_vocabsize, assays_vocabsize, latentdim, lbldim)
        
    def forward(self, smilesinput, labelsinput, valueinput):
        # smilesinput, labelsinput, valueinput = trn[1:5][0], trn[1:5][1], trn[1:5][2]
        
        zMean, zLogvar, embedlbl = self.encoder(smilesinput, labelsinput, valueinput)
        std = torch.exp(0.5*zLogvar)
        eps = torch.randn_like(std)
        
        encodedinput = zMean + eps*std
        encodedinput = self.dropout(encodedinput)
        
        yPred = self.decoder(encodedinput, embedlbl, valueinput)
        return yPred, zMean, zLogvar