import torch
import torch.nn as nn
import torch.nn.functional as F
import pathlib
import cvae.models.tokenizer

class PredVAE(nn.Module):
    
    def __init__(self, len_charset, nlabels, verbose=False):
        super(PredVAE, self).__init__()
        
        ldim = 292
        
        self.embed_label = nn.Embedding(nlabels, 100)
        self.encoder = nn.Sequential(
            nn.Conv1d(len_charset, 10, kernel_size=9),
            nn.ReLU(),
            
            nn.Conv1d(10, 9, kernel_size=9),
            nn.ReLU(),
            
            nn.Conv1d(9, 10, kernel_size=11),
            nn.ReLU(),
        )
        
        self.fcencoder = nn.Sequential(
            nn.Linear(940, 940), nn.ReLU(), nn.BatchNorm1d(940),
            nn.Linear(940, 500), nn.ReLU(), nn.BatchNorm1d(500),
        )
        
        self.linear_mean = nn.Linear(600, ldim)
        self.linear_logvar = nn.Linear(600, ldim)
        
        self.dec_seq = nn.LSTM(ldim, 250, 3, batch_first=True, bidirectional=True)
        self.dec_sig = nn.Sequential(nn.Linear(500, len_charset), nn.Sigmoid())
        
        self.valdecoder = nn.Sequential(
            nn.Linear(ldim,ldim),nn.BatchNorm1d(ldim),nn.ReLU(),
            nn.Linear(ldim,ldim),nn.BatchNorm1d(ldim),nn.ReLU(),
            nn.Linear(ldim,100),nn.ReLU(),
            nn.Linear(100,1), nn.Sigmoid())
        
    def encode(self, insmi, inlbl):
        elbl = self.embed_label(inlbl)
        esmi = self.encoder(insmi)
        esmi = esmi.view(esmi.shape[0], -1)
        esmi = self.fcencoder(esmi)
        
        ecat = torch.cat([esmi,elbl], dim=1)
        zmean = self.linear_mean(ecat)
        zlogvar = self.linear_logvar(ecat)
        return zmean, zlogvar
    
    def sample(self, zmean, zlogvar):
        epsilon = 0.1*torch.randn_like(zlogvar)
        std = torch.exp(0.5 * zlogvar)
        z = std * epsilon + zmean
        return z
    
    def decode(self, z):
        zrep = z.unsqueeze(1).repeat(1, 120, 1)
        dec = self.dec_seq(zrep)[0]
        dec = self.dec_sig(dec)
        val = self.valdecoder(z)
        return dec.permute(0,2,1), val
    
    def forward(self, insmi, inlbl):
        zmean, zlogvar = self.encode(insmi, inlbl)
        z = self.sample(zmean, zlogvar)
        decsmi, decval = self.decode(z)
        return decsmi, decval, zmean, zlogvar
    
    def loss(self, decsmi, decval, insmi, inval, zmean, zlogvar):
        
        recon_loss = F.binary_cross_entropy(decsmi, insmi, reduction='sum')
        
        Bkl = 1
        kl_loss = -0.5 * torch.sum(1 + zlogvar - zmean.pow(2) - zlogvar.exp())
        kl_loss = Bkl * kl_loss
        
        Bval = 10
        inval = inval.unsqueeze(1)
        val_loss = Bval*F.binary_cross_entropy(decval, inval, reduction='sum')
        
        return recon_loss + kl_loss + val_loss, recon_loss, kl_loss, val_loss
    
    @staticmethod
    def load(path = pathlib.Path("brick/pvae.pt")):
        state = torch.load(path)
        nlbls = state['embed_label.weight'].shape[0] # number of labels
        nchrs = state['encoder.0.weight'].shape[1] # number of characters
        mod = PredVAE(nchrs, nlbls)
        mod.load_state_dict(state)
        return mod

# def recon_acc(self, decsmi, insmi):
#     rdecoded = torch.round(decsmi)
#     truepositive = torch.sum(rdecoded * insmi)
#     truenegative = torch.sum((1 - rdecoded) * (1 - insmi))
#     positive = torch.sum(inchem)
#     negative = torch.sum(1 - inchem)
    
#     se = truepositive / positive
#     sp = truenegative / negative
#     return 100.0 * (se + sp)/2.0

# def valacc(self, decval, inval):
#     pval = torch.round(decval)
#     tp = torch.sum(pval * inval)
#     tn = torch.sum((1 - pval) * (1 - inval))
#     pos,neg = torch.sum(inval), torch.sum(1 - inval)
    
#     se, sp = tp/pos, tn/neg
#     return 100.0 * (se + sp)/2.0
    
# def rand(self, n):
#     smi = 'C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O'
#     insmi = torch.Tensor(cvae.models.tokenizer.smiles_one_hot(smi)).unsqueeze(0).to(device)
#     z = self.encode(insmi, torch.Tensor([0]).long().to(device))[0]
#     z = z.repeat(100,1)
#     z = z + 0.2*torch.randn_like(z)
#     decsmi = self.decode(z)[0].permute(0,2,1)
#     argmax = torch.argmax(decsmi,dim=2).cpu().numpy()
#     osmi = [cvae.models.tokenizer.decode_smiles_from_indexes(x) for x in argmax]
#     print("\n".join(osmi))
    
#     def isvalid(smi): 
#         try:
#             return cvae.models.tokenizer.valid_smiles(smi)
#         except:
#             return False
    
#     vsmi = [x for x in osmi if isvalid(x)]
    
#     return decsmi

# def sample_mol(self, smiles, sample_coeff=0.1):
#     insmi = cvae.models.tokenizer.smiles_one_hot(smiles)
#     zmean, zlogvar = self.encode(insmi)
    
#     epsilon = sample_coeff*torch.randn_like(zlogvar)
#     std = torch.exp(0.5 * zlogvar)
#     z = std * epsilon + zmean
    
#     z = self.sample(zmean, zlogvar)
#     decsmi, _ = self.decode(z)
#     return decsmi