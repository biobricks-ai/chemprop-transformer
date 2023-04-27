import json, os
import torch

import utils, CVAE.model

class BioSim:
    
    def __init__(self, model, smiles_vocab, assays_vocab, model_info):
        self.model = model
        self.smiles_vocab = smiles_vocab
        self.assays_vocab = assays_vocab
        self.model_info = model_info
        self.smiles_padlength = model_info['smiles_padlength']
        
    @staticmethod
    def load():
        modinfo = utils.loadparams()
        modinfo.update(json.loads(utils.res("data/processed/modelInfo.json").read_text()))
        modinfo.update(torch.load(utils.res("models/train/bestModel.pt")))
        mod = CVAE.model.CVAE(
            modinfo["smiles_padlength"], 
            modinfo["smiles_vocabsize"], 
            modinfo["assays_vocabsize"], 
            modinfo['latentDim'])
        
        vocabs = json.loads(utils.res("data/processed/vocabs.json").read_text())
        smiles_vocab = vocabs['smiles_vocab']
        assays_vocab = vocabs['assays_vocab']
        
        return BioSim(mod, smiles_vocab, assays_vocab, modinfo)
        
    def smiles_to_tensor(self, smiles):    
        "transform smiles to index tensor by using smiles_vocab"
        padlength = self.smiles_padlength
        padsmiles = smiles.ljust(padlength)[0:padlength]
        return [self.smiles_vocab[c] for c in padsmiles]
    
    def tensor_to_smiles(self, tensor):
        smivocab = {v:k for k,v in self.smiles_vocab.items()}
        return ''.join([smivocab[i.item()] for i in tensor[0]])
        
    def predict(self, model, smiles):
        smiles = "CCOCN(C(=O)CCl)c1c(C)cccc1CC"
        in_smi = torch.tensor([self.smiles_to_tensor(smiles)])
        in_ass = torch.tensor([self.assays_vocab['733a9c0b-fdcc-4c98-9525-e052302d6afa']])
        
        res = self.model(in_smi,in_ass,torch.tensor([0])) # 1, 244, 67
        ten = torch.argmax(res[0],dim=-1)[0]
        ''.join([smivocab[i.item()] for i in ten])
        # argmax of res on 3rd rank

class ChemVecStore:
    "faiss db for chemical, assay, value embeddings"
    
    def __init__(self, faiss):
        self.faiss = faiss
        
    @staticmethod
    def build():
        "load chemharmony, build vectorizer, chemical, assay, value, embedding store"
        
        
class Evaluator:
    "class for functions to evaluate chemical similarity"