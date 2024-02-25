import pandas as pd, tqdm, sklearn.metrics
import torch
import cvae.tokenizer, cvae.models.multitask_transformer as mt, cvae.utils

DEVICE = torch.device(f'cuda:0')
tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('data/processed/selfies_property_val_tokenizer')
model = mt.MultitaskDecoderTransformer.load("brick/working_mtt").to(DEVICE)

# EVALUATION FUNCTIONS ===================================================================

assay_indexes = torch.tensor(list(tokenizer.assay_indexes().values()), device=DEVICE)
index_assays = {v: k for k, v in tokenizer.assay_indexes().items()}
value_indexes = torch.tensor(list(tokenizer.value_indexes().values()), device= DEVICE)
index_values = {v: k for k, v in tokenizer.value_indexes().items()}

def extract_ordered_assays(tensor,assay_indexes,index_assays):
    mask = torch.isin(tensor, assay_indexes)
    return [index_assays[item.item()] for item in tensor[mask]]

def extract_ordered_values(tensor,value_indexes,index_values):
    mask = torch.isin(tensor, value_indexes)
    return [index_values[item.item()] for item in tensor[mask]]

def extract_probabilities_of_one(out, probs, value_indexes):
    idx = torch.nonzero(torch.isin(out, value_indexes),as_tuple=True)
    value_probs = probs[idx[0].unsqueeze(1), idx[1].unsqueeze(1), value_indexes]
    sum_val = value_probs.sum(dim=1, keepdim=True)
    normalized_value_probs = value_probs / sum_val
    return normalized_value_probs[:, 1]

def generate_position_tensors(out, value_indexes):
    num_props = torch.sum(torch.isin(out, value_indexes), dim=1)
    return torch.cat([torch.arange(size.item()) for size in num_props])

# EVALUATION LOOP ===================================================================

tst = mt.SequenceShiftDataset("data/processed/multitask_tensors/tst", tokenizer.pad_idx, tokenizer.SEP_IDX, tokenizer.END_IDX)
tstdl = torch.utils.data.DataLoader(tst, batch_size=2048, shuffle=False)
out_df = pd.DataFrame()

for j in range(2):
    for i, (inp, teach, out) in tqdm.tqdm(enumerate(tstdl), total=len(tstdl)):
        inp, teach, out = inp.to(DEVICE), teach.to(DEVICE), out.to(DEVICE)
        x = torch.gt(torch.sum(torch.isin(out, value_indexes),dim=1),9)
        inp_x,teach_x, out_x = inp[x],teach[x],out[x]
        prob = torch.softmax(model(inp_x, teach_x),dim=2).detach()
        probs = extract_probabilities_of_one(out_x, prob, value_indexes).cpu().numpy()
        assays = extract_ordered_assays(out_x, assay_indexes, index_assays)
        values = extract_ordered_values(out_x, value_indexes, index_values)
        position = generate_position_tensors(out_x, value_indexes).cpu().numpy().tolist()
        batch_df = pd.DataFrame({ 'batch': i, 'assay': assays, 'value': values, 'probs':probs, 'position':position })
        out_df = pd.concat([out_df, batch_df], ignore_index=True)
        
# GENERATE STRATIFIED EVALUATIONS FOR POSITION 0-9 ===============================

assay_metrics = []
grouped = out_df.groupby(['position','assay'])

for (position,assay), group in tqdm.tqdm(grouped):
    y_true, y_pred = group['value'].values, group['probs'].values
    if sum(y_true==0) < 100 or sum(y_true==1) < 100 : continue
    auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
    acc = sklearn.metrics.accuracy_score(y_true, y_pred > 0.5)
    bac = sklearn.metrics.balanced_accuracy_score(y_true, y_pred > 0.5)
    assay_metrics.append({'position': position, 'assay': assay, 'AUC': auc, 'ACC': acc, 'BAC': bac, "NUM_POS": sum(y_true==1), "NUM_NEG": sum(y_true==0)})

metrics_df = pd.DataFrame(assay_metrics)
metrics_df.sort_values(by=['AUC'], inplace=True, ascending=False)
metrics_df.to_csv("metrics.csv")

position_df = metrics_df.groupby('position').aggregate({'AUC': 'median', 'ACC': 'median', 'BAC': 'median'})
position_df.sort_values(by=['AUC'], inplace=True, ascending=False)
position_df.to_csv("position_metrics.csv")
auc, acc, bac = metrics_df['AUC'].median(), metrics_df['ACC'].median(), metrics_df['BAC'].median()
print(f"Metrics over position:\tAUC: {auc:.4f}\tACC: {acc:.4f}\tBAC: {bac:.4f}")