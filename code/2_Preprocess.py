import json, torch, numpy as np, pandas as pd
import utils, pathlib
from tqdm import tqdm
from torch.utils.data import TensorDataset, random_split
import models, models.tokenizer
tqdm.pandas()


params = utils.loadparams()['preprocessing']
data = pd.read_csv(utils.res(params["rawDataPath"]))[['smiles','assay','value']]
data = data.sample(frac=1) # 6738282 rows

# REMOVE ROWS THAT CAN'T BE TOKENIZED =======================================
charset = models.tokenizer.charset
valid_smiles = lambda smiles: all(char in charset for char in smiles)
valid_smiles = data['smiles'].progress_apply(valid_smiles)
data = data[valid_smiles] # 6733185 rows

# ASSAY UUID TO INDEX TENSOR called `tass` or 'tensor_assay' 
assayidx = {assay:i for i,assay in enumerate( np.unique(data['assay']) )}
tass = data['assay'].progress_apply(lambda a: assayidx[a])
tass = torch.tensor(tass)

# SMILES TO RANK 2 INDEX TENSOR called `tsmi` or `tensor_smiles`
tsmi = data['smiles'].progress_apply(models.tokenizer.smiles_one_hot)
tsmi = torch.vstack(tsmi)

# BINARY VALUE TO TENSOR `tval` or `tensor_value`
tval = torch.tensor(data['value'])

# MolecularVAE ENCODINGS
mvae = models.MoleculeVAE.load(utils.res("resources/chembl_23_model.h5"))

# WRITE MODEL PARAMS AND VOCABS TO FILE
outdir = pathlib.Path(utils.res(params["outDataFolder"]))
outdir.mkdir(parents=True, exist_ok=True)

vocabs = {'smiles_vocab': smiCharToInt, 'assays_vocab': assayidx}
(outdir / 'vocabs.json').write_text(json.dumps(vocabs, indent=4))

model_info = { 'smiles_padlength': smi_padlength }
model_info['smiles_vocabsize'] = len(smiCharToInt)
model_info['assays_vocabsize'] = len(assayidx)
(outdir / 'modelInfo.json').write_text(json.dumps(model_info, indent=4))

# BUILD TRAIN/TEST/VALIDATION 
tdata = TensorDataset(tsmi, tass, tval)
ntrn = int(0.8*len(tdata))
nval = (len(tdata) - ntrn) // 2  
nhld = len(tdata) - ntrn - nval 

trn, val, hld = random_split(tdata, [ntrn, nval, nhld])
torch.save(trn, outdir / 'train.pt')
torch.save(val, outdir / 'validation.pt')
torch.save(hld, outdir / 'holdout.pt')