import json, torch, numpy as np, pandas as pd
import cvae.utils as utils, pathlib
from tqdm import tqdm
from torch.utils.data import TensorDataset, Subset
import cvae.models, cvae.models.tokenizer as tokenizer
tqdm.pandas()


params = utils.loadparams()['preprocessing']
data = pd.read_csv(utils.res(params["rawDataPath"]))[['smiles','assay','value']]
data = data.sample(frac=1) # 6738282 rows
data = data.head(100000) # 1000000 rows

# REMOVE ROWS WITH ASSAYS THAT HAVE LESS THAN 100 `0` and `1` values=======
zero_counts = data[data['value'] == 0].groupby('assay').size().reset_index(name='zero_count')
valid_assays = zero_counts[zero_counts['zero_count'] >= 100]['assay']
data = data[data['assay'].isin(valid_assays)]

one_counts = data[data['value'] == 1].groupby('assay').size().reset_index(name='one_count')
valid_assays = one_counts[one_counts['one_count'] >= 100]['assay']
data = data[data['assay'].isin(valid_assays)]

# REMOVE ROWS WITH INVALID SMILES ===========================================
# must have characters in tokenizer charset, pass rdkit, and length < 121
valid_smiles = data['smiles'].progress_apply(tokenizer.valid_smiles)
data = data[valid_smiles] # 6733185 rows

# TRANSFORM TO TENSOR DATASET ===============================================
# ASSAY UUID TO INDEX TENSOR called `tass` or 'tensor_assay' 
assayidx = {assay:i for i,assay in enumerate( np.unique(data['assay']) )}
tass = data['assay'].progress_apply(lambda a: assayidx[a]).to_numpy()
tass = torch.tensor(tass)

# SMILES TO RANK 2 INDEX TENSOR called `tsmi` or `tensor_smiles`
tsmi = data['smiles'].progress_apply(tokenizer.smiles_one_hot).to_numpy()
tsmi = np.stack(tsmi)
tsmi = torch.tensor(tsmi)

# BINARY VALUE TO TENSOR `tval` or `tensor_value`
tval = torch.tensor(data['value'].to_numpy())

# MACCS EMBEDDINGS
tmorgan = data['smiles'].progress_apply(tokenizer.smiles_to_morgan_fingerprint)
tmorgan = np.stack(tmorgan)
tmorgan = torch.tensor(tmorgan)

# MolecularVAE ENCODINGS
mvae = cvae.models.mvae.load(utils.res("resources/chembl_23_model_pytorch.pt"))
tmvae = mvae(tsmi)[0].detach()

# BUILD TRAIN/TEST/VALIDATION 
ntot = len(tsmi)
ntrn = int(0.8*ntot)
nval = (ntot - ntrn) // 2  
nhld = ntot - ntrn - nval 

# create a tensor flagging whether the row is trn, val or hld
ttrnflag = torch.zeros(ntot, dtype=torch.long)
ttrnflag[ntrn:ntrn + nval] = 1
ttrnflag[ntrn + nval:] = 2

def tslice(td, start, end):
    tensors = [t[start:end] for t in td.tensors]
    return TensorDataset(*tensors)

td = TensorDataset(tsmi, tass, tval, tmorgan, tmvae, ttrnflag)
trn, val, hld = tslice(td,0,ntrn), tslice(td,ntrn,ntrn + nval), tslice(td,ntrn + nval, ntot)
hld.tensors[5].unique(return_counts=True)

outdir = pathlib.Path(utils.res('data/processed'))
torch.save(trn, outdir / 'train.pt')
torch.save(val, outdir / 'validation.pt')
torch.save(hld, outdir / 'holdout.pt')

# WRITE MODEL PARAMS AND VOCABS TO FILE
# count assays

# outdir.mkdir(parents=True, exist_ok=True)

# vocabs = {'smiles_vocab': smiCharToInt, 'assays_vocab': assayidx}
# (outdir / 'vocabs.json').write_text(json.dumps(vocabs, indent=4))

# model_info = { 'smiles_padlength': smi_padlength }
# model_info['smiles_vocabsize'] = len(smiCharToInt)
# model_info['assays_vocabsize'] = len(assayidx)
# (outdir / 'modelInfo.json').write_text(json.dumps(model_info, indent=4))

