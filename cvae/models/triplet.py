import torch
import torch.nn as nn
import torch.nn.functional as F


class IsometricEmbedding(nn.Module):
    
    def __init__(self, numlabel):
        super(IsometricEmbedding, self).__init__()

        self.embedlbl = nn.Sequential(nn.Embedding(numlabel, 10),nn.ReLU())
        self.embedlbl[0].weight.data = torch.randn(numlabel, 10)*0.01
        
        def isolayer():
            layer = nn.Linear(2048, 2048)
            layer.weight.data = torch.eye(2048) + torch.randn(2048, 2048)*0.01
            layer.bias.data = torch.zeros(2048)
            return layer
        
        self.embed = nn.Sequential(
            nn.Linear(2058, 2048),nn.ReLU(),
            isolayer(),nn.ReLU(),
            isolayer(),nn.ReLU())
        
        initweight = torch.eye(2048) + torch.randn(2048, 2048)*0.01
        initweight = torch.cat([initweight, torch.randn(2048, 10)], dim=1)
        self.embed[0].weight.data = initweight
        self.embed[0].bias.data = torch.zeros(2048)
        
    def forward(self, inp, lbl):
        cat = torch.cat([inp, self.embedlbl(lbl)], dim=1)
        cat = self.embed(cat)
        emb = inp * cat
        emb = F.normalize(emb, dim=1)
        return emb
    
    def loss(self, inp, out):
        ip_inp = 1-torch.matmul(inp, inp.transpose(0, 1))
        ip_out = 1-torch.matmul(out, out.transpose(0, 1))
        loss = torch.sum(torch.pow(ip_inp - ip_out, 2))
        return loss
        
    
    @staticmethod
    def load(path = "brick/morganisoembed.pt"):
        state = torch.load(path)
        model = IsometricEmbedding(numlabel=state["embedlbl.0.weight"].shape[0])
        model.load_state_dict(state)
        return model

class TripletModel(nn.Module):
    
    def __init__(self, numlabel):
        super(TripletModel, self).__init__()
        self.isoembed = IsometricEmbedding(numlabel)

    def forward(self, lbl, anc, pos, neg):
            
        embed = lambda x, l: self.isoembed(x, l)
        embanc = embed(anc, lbl)
        embpos = embed(pos, lbl)
        embneg = embed(neg, lbl)

        return embanc, embpos, embneg
    
    def loss(self, anc, embanc, pos, embpos, neg, embneg):
        
        isoancloss = torch.sqrt(self.isoembed.loss(anc, embanc))
        isoposloss = torch.sqrt(self.isoembed.loss(pos, embpos))
        isonegloss = torch.sqrt(self.isoembed.loss(neg, embneg))
        isoloss = (isoancloss + isoposloss + isonegloss)/3.
        
        ancposip = (1.+F.cosine_similarity(embanc, embpos))/2. # 0 to 1
        ancnegip = (1.+F.cosine_similarity(embanc, embneg))/2. # 0 to 1
        diff = ancnegip - ancposip  + 1. # 0 to 2 
        tripleloss = torch.sum(diff)
        
        return tripleloss + 0.1*isoloss, tripleloss, isoloss
