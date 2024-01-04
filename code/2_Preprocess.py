# DEPRECATED: This file is deprecated. 
# import json, torch, numpy as np, pandas as pd, os, sys
# sys.path.insert(0, os.getcwd())
# import cvae.utils as utils, pathlib
# from tqdm import tqdm
# from torch.utils.data import TensorDataset, Subset
# import cvae.models, cvae.models.tokenizer as tokenizer
# tqdm.pandas()


# params = utils.loadparams()['preprocessing']
# data = pd.read_csv(utils.res(params["rawDataPath"]))[['smiles','assay','value']]
# data = data.sample(frac=1.0) # 15 352 907 rows

# # REMOVE ROWS WITH ASSAYS THAT HAVE LESS THAN 100 `0` and `1` values=======
# zero_counts = data[data['value'] == 0].groupby('assay').size().reset_index(name='zero_count')
# valid_assays = zero_counts[zero_counts['zero_count'] >= 500]['assay']
# data = data[data['assay'].isin(valid_assays)]

# one_counts = data[data['value'] == 1].groupby('assay').size().reset_index(name='one_count')
# valid_assays = one_counts[one_counts['one_count'] >= 500]['assay']
# data = data[data['assay'].isin(valid_assays)] # 11 451 538 rows

# # REMOVE ASSAYS WITH A CLASS IMBALANCE GREATER THAN 10% ====================
# # Calculate class imbalance for each assay
# assay_counts = data.groupby(['assay', 'value']).size().unstack(fill_value=0).reset_index()
# assay_counts['imbalance'] = abs(assay_counts[0] - assay_counts[1]) / (assay_counts[0] + assay_counts[1])

# # Filter out assays with imbalance greater than 10%
# balanced_assays = assay_counts[(assay_counts['imbalance'] <= 0.1) | (assay_counts['imbalance'] >= 0.9)]['assay']
# data = data[data['assay'].isin(balanced_assays)]  # 8 229 345 rows

# # REMOVE ROWS WITH INVALID SMILES ===========================================
# # must have characters in tokenizer charset, pass rdkit, and length < 121
# valid_smiles = data['smiles'].progress_apply(tokenizer.valid_smiles)
# data = data[valid_smiles] # 8 168 977 rows

# # TRANSFORM TO TENSOR DATASET ===============================================
# # ASSAY UUID TO INDEX TENSOR called `tass` or 'tensor_assay' 
# assayidx = {assay:i for i,assay in enumerate( np.unique(data['assay']) )}
# tass = data['assay'].progress_apply(lambda a: assayidx[a]).to_numpy()
# tass = torch.tensor(tass)

# # SMILES TO RANK 2 INDEX TENSOR called `tsmi` or `tensor_smiles`
# tsmi = np.stack(data['smiles'].progress_apply(tokenizer.smiles_one_hot).to_numpy())
# tsmi = torch.tensor(tsmi)

# # BINARY VALUE TO TENSOR `tval` or `tensor_value`
# tval = torch.tensor(data['value'].to_numpy())

# # MACCS EMBEDDINGS
# tmorgan = np.stack(data['smiles'].progress_apply(tokenizer.smiles_to_morgan_fingerprint))
# tmorgan = torch.tensor(tmorgan)

# # BUILD TRAIN/TEST/VALIDATION 
# ntot = len(tsmi)
# ntrn = int(0.8*ntot)
# nval = (ntot - ntrn) // 2  
# nhld = ntot - ntrn - nval 

# # create a tensor flagging whether the row is trn, val or hld
# ttrnflag = torch.zeros(ntot, dtype=torch.long)
# ttrnflag[ntrn:ntrn + nval] = 1
# ttrnflag[ntrn + nval:] = 2

# def tslice(td, start, end):
#     tensors = [t[start:end] for t in td.tensors]
#     return TensorDataset(*tensors)

# td = TensorDataset(tsmi, tass, tval, tmorgan, ttrnflag)
# trn, val, hld = tslice(td,0,ntrn), tslice(td,ntrn,ntrn + nval), tslice(td,ntrn + nval, ntot)

# outdir = pathlib.Path(utils.res('data/processed'))
# outdir.mkdir(parents=True, exist_ok=True)
# torch.save(td, outdir / 'data.pt')
# torch.save(trn, outdir / 'train.pt')
# torch.save(val, outdir / 'validation.pt')
# torch.save(hld, outdir / 'holdout.pt')

# # GRAPH EMBEDDINGS
