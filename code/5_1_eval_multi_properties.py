import itertools
import pandas as pd, tqdm, sklearn.metrics, torch, numpy as np, os
import cvae.tokenizer, cvae.models.multitask_transformer as mt, cvae.utils, cvae.models.mixture_experts as me

DEVICE = torch.device(f'cuda:0')
outdir = cvae.utils.mk_empty_directory("data/metrics", overwrite=True)
tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')
# model = mt.MultitaskTransformer.load("brick/mtransform3_nprops1").to(DEVICE)
model = me.MoE.load("brick/moe").to(DEVICE)
model = torch.nn.DataParallel(model)

# EVALUATION LOOP ===================================================================
assay_indexes = torch.tensor(list(tokenizer.assay_indexes().values()), device=DEVICE)
value_indexes = torch.tensor(list(tokenizer.value_indexes().values()), device= DEVICE)

def run_eval(i, raw_inp, raw_out, out_df, nprops):
    inp, raw_out = raw_inp.to(DEVICE), raw_out.to(DEVICE)
        
    # filter to instances with at least nprops properties
    x = torch.greater_equal(torch.sum(torch.isin(raw_out, value_indexes),dim=1),nprops)
    chemical_id = torch.where(x)[0] + (i * batch_size)
    inp, trunc_out = inp[x], raw_out[x,1:(2*nprops + 1)].reshape(-1,nprops,2)
    
    # if all of x is false skip
    if len(chemical_id) == 0: 
        return out_df
    
    # get all permutations
    perm_indices = list(itertools.permutations(range(nprops)))
    perm_out = torch.cat([trunc_out[:, list(perm), :] for perm in perm_indices],dim=0).reshape(-1,nprops*2)
    sep_tensor = torch.full((perm_out.size(0),1), tokenizer.SEP_IDX, device=raw_out.device)
    zer_tensor = torch.zeros_like(sep_tensor, device=raw_out.device)
    out = torch.cat([sep_tensor,perm_out,zer_tensor],dim=1)
    
    # make teach tensor
    one_tensor = torch.ones_like(sep_tensor, device=out.device)
    teach = torch.cat([one_tensor, out[:,:-1]], dim=1)
    
    # repeat interleave input for all the permutations. if inp has idxs 1,2 then the below gives us 1,1,2,2
    rep_inp = inp.repeat(len(perm_indices),1)
    
    # get model predictions as a prob
    prob = torch.softmax(model(rep_inp, teach),dim=2).detach()
    
    # get out assays and the assay with the highest prob
    assays = out[torch.isin(out, assay_indexes)].cpu().numpy()
    prob_assays = torch.argmax(prob, dim=2)[torch.isin(out, assay_indexes)].cpu().numpy()
    
    # get out values and the value with the highest prob and the prob of the `1`` value
    values = out[torch.isin(out, value_indexes)].cpu().numpy()
    
    probmax_vals = torch.argmax(prob, dim=2)[torch.isin(out, value_indexes)].cpu().numpy()
    rawprobs = prob[torch.isin(out, value_indexes)][:,value_indexes]
    probs = (rawprobs / rawprobs.sum(dim=1, keepdim=True))[:,1].cpu().numpy()
    
    # get position of each value in the out tensor
    num_props = torch.sum(torch.isin(out, assay_indexes), dim=1)
    position = torch.cat([torch.arange(size.item()) for size in num_props]).cpu().numpy()
    
    # repeat chemical_id 10x
    chemical_id = torch.repeat_interleave(chemical_id, len(perm_indices))
    chemical_id = torch.repeat_interleave(chemical_id, num_props).cpu().numpy()
    
    # cut assays up into groups of nprops then build 10 strings with assay 0, assay 0 + assay 1, assay 0 + assay 1 + assay 2, etc.
    assays_reshaped = assays.reshape(-1, nprops).astype(str)
    prior_assays = [' + '.join(assays_reshaped[i, :j+1]) for i in range(len(assays_reshaped)) for j in range(nprops)]
    batch_df = pd.DataFrame({'batch': i, 'chemical_id': chemical_id, 
                                'prior_assays': prior_assays,
                                'assay': assays, 
                                'value': values, 'probs':probs, 'nprops':position,
                                'prob_assays': prob_assays, 'prob_vals': probmax_vals})
    
    return pd.concat([out_df, batch_df]) if len(out_df) > 0 else batch_df

batch_size=10
nprops = 5
val = mt.SequenceShiftDataset("data/tensordataset/multitask_tensors/hld", tokenizer, nprops=nprops)
valdl = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False)
out_df = pd.DataFrame({'chemical_id':[], 'prior_assays':[], 'assay':[], 'value':[], 'probs':[], 'nprops':[], 'prob_assays':[], 'prob_vals':[]})
for _ in range(100):
    for i, (raw_inp, _, raw_out) in tqdm.tqdm(enumerate(valdl), total=len(val)/batch_size):
        out_df = run_eval(i, raw_inp, raw_out, out_df, nprops)
    out_df.drop_duplicates(subset=['chemical_id', 'prior_assays'],inplace=True)

sum(out_df['assay'] == out_df['prob_assays']) / len(out_df)
sum(out_df['value'] == out_df['prob_vals']) / len(out_df)

# GENERATE STRATIFIED EVALUATIONS FOR POSITION 0-9 ===============================
assay_metrics = []
grouped = out_df.groupby(['nprops','assay'])
for (position,assay), group in tqdm.tqdm(grouped):
    y_true, y_pred = group['value'].values, group['probs'].values
    y_true = np.array([0 if x == 6170 else 1 for x in y_true])
    nchem = len(group['chemical_id'].unique())
    if sum(y_true==0) < 10 or sum(y_true==1) < 10 or nchem < 20 : continue
    assay_metrics.append({
        'nprops': position, 
        'assay': assay, 
        'AUC': sklearn.metrics.roc_auc_score(y_true, y_pred), 
        'ACC': sklearn.metrics.accuracy_score(y_true, y_pred > 0.5),
        'BAC': sklearn.metrics.balanced_accuracy_score(y_true, y_pred > 0.5),
        'cross_entropy_loss': sklearn.metrics.log_loss(y_true, y_pred),
        "NUM_POS": sum(y_true==1), 
        "NUM_NEG": sum(y_true==0)})

metrics_df = pd.DataFrame(assay_metrics)
metrics_df.sort_values(by=['AUC'], inplace=True, ascending=False)
metrics_df.to_csv("data/metrics/multitask_metrics.csv", index=False)