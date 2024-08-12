import itertools, uuid, shutil
import pandas as pd, tqdm, sklearn.metrics, torch, numpy as np, os
import cvae.tokenizer, cvae.models.multitask_transformer as mt, cvae.utils, cvae.models.mixture_experts as me

DEVICE = torch.device(f'cuda:0')
outdir = cvae.utils.mk_empty_directory("data/finetune/metrics", overwrite=True)


model = me.MoE.load("data/finetune/model").to(DEVICE)
tokenizer = model.tokenizer
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
    values_reshaped = values.reshape(-1, nprops).astype(str)
    prior_assays = [' + '.join(assays_reshaped[i, :j+1]) for i in range(len(assays_reshaped)) for j in range(nprops)]
    prior_values = [values_reshaped[i, :j+1] for i in range(len(values_reshaped)) for j in range(nprops)]
    batch_df = pd.DataFrame({'batch': i, 'chemical_id': chemical_id, 
                                'prior_assays': prior_assays, 'prior_values': prior_values,
                                'assay': assays, 
                                'value': values, 'probs':probs, 'nprops':position,
                                'prob_assays': prob_assays, 'prob_vals': probmax_vals})
    
    return pd.concat([out_df, batch_df]) if len(out_df) > 0 else batch_df

batch_size = 100
nprops = 1
val = mt.SequenceShiftDataset("data/finetune/multitask_tensors/hld", tokenizer, nprops=nprops)
valdl = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False)
out_df = pd.DataFrame({'chemical_id':[], 'prior_assays':[], 'prior_values':[], 'assay':[], 'value':[], 'probs':[], 'nprops':[], 'prob_assays':[], 'prob_vals':[]})

# create tempdir and ensure it's empty
temp_dir = cvae.utils.mk_empty_directory("data/finetune/metrics/temp")
for epoch in tqdm.tqdm(range(100)):
    for i, (raw_inp, _, raw_out) in tqdm.tqdm(enumerate(valdl), total=len(val)/batch_size):
        out_df = run_eval(i, raw_inp, raw_out, out_df, nprops)

    out_df.drop_duplicates(subset=['chemical_id', 'prior_assays'],inplace=True)
    out_df.to_csv(f"data/finetune/metrics/temp/multitask_predictions_{str(uuid.uuid4())}.csv", index=False)


# read all the temp files and concatenate them
out_df = pd.concat([pd.read_csv(f"data/finetune/metrics/temp/{x}") for x in os.listdir("data/finetune/metrics/temp")])
out_df.drop_duplicates(subset=['chemical_id', 'prior_assays'],inplace=True)
out_df['prior_assays'] = out_df['prior_assays'].apply(lambda x: x.split(' + '))

out_df.to_csv("data/finetune/metrics/multitask_predictions.csv", index=False)

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
metrics_df.to_csv("data/finetune/metrics/multitask_metrics.csv", index=False)

# BUILD FIGURES ======================================================================
import matplotlib.pyplot as plt, pandas as pd, seaborn as sns
import random

# SETUP =================================================================================
df = pd.read_csv('data/finetune/metrics/multitask_metrics.csv')
df.aggregate({'AUC': 'median', 'ACC': 'median', 'BAC': 'median', "cross_entropy_loss": 'median'})

df[(df['NUM_POS'] > 20) & (df['NUM_NEG'] >= 20)]\
    .groupby('nprops')\
    .aggregate({'AUC': 'median', 'ACC': 'median', 'BAC': 'median', "cross_entropy_loss": 'median', 'assay': 'nunique'})

# AUC HISTOGRAM =========================================================================
def auc_histogram(df,nprops):
    assays = df.groupby('assay').filter(lambda x: x['nprops'].nunique() == nprops)['assay'].unique()
    plotdf = df[df['nprops'].isin(list(range(nprops))) & df['assay'].isin(assays)]

    plt.style.use('dark_background')
    g = sns.FacetGrid(plotdf, col='nprops', height=5, aspect=1.3, col_wrap=3, sharex=False, sharey=False)
    g.map(plt.hist, 'AUC', bins=30, alpha=1.0, edgecolor='white', linewidth=1, color='#16A085')

    for ax, nprop in zip(g.axes.flatten(), [0, 1, 2, 3, 4]):
        median_auc = plotdf[plotdf['nprops'] == nprop]['AUC'].median()
        ax.axvline(median_auc, color='red', linestyle='dashed', linewidth=3)
        ax.annotate(f'Median AUC: {median_auc:.3f}', xy=(median_auc-0.125, 0.95), xycoords='axes fraction',
                    fontsize=12, color='red', fontweight='bold', ha='right', va='top')
        ax.set_title(f'Prior Properties = {nprop}', color='white', fontsize=18)
        ax.set_xlabel('AUC', color='white', fontsize=14)
        ax.set_ylabel('Number of Properties', color='white', fontsize=14)
        ax.tick_params(colors='white', labelsize=12)

    g.figure.suptitle('Histogram of AUC per Property', color='white', fontsize=22, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.9)  # Adjust this value as needed to make room for the title
    g.figure.tight_layout(rect=[0, 0.03, 1, 0.9])  # Adjust the rect to ensure title is visible
    g.add_legend()
    plt.savefig('notebook/plots/finetune/multitask_transformer_metrics.png', facecolor='none', transparent=True)

auc_histogram(df[(df['NUM_POS'] > 20) & (df['NUM_NEG'] >= 20)], nprops=1)


# AUC BY POSITION =======================================================================
## select assays that appear with all values of nprops

assays = df.groupby('assay').filter(lambda x: x['nprops'].nunique() == 5)['assay'].unique()
posdf = df[df['assay'].isin(assays)]
posdf.sort_values(by=['assay', 'nprops'], inplace=True)

# Get the median AUC by nprops
median_df = posdf.groupby(['nprops']).agg({'AUC': 'median'}).reset_index()

# Randomly select up to 1000 assays
sampled_assays = random.sample(list(assays), min(1000, len(assays)))
sampled_df = posdf[posdf['assay'].isin(sampled_assays)]

plt.style.use('dark_background')
plt.figure(figsize=(12, 6))

# Create line plot for randomly selected assays
sns.lineplot(data=sampled_df, x='nprops', y='AUC', units='assay', estimator=None, linewidth=0.5, alpha=0.3, color='gray')

# Create line plot for median AUC with points
sns.lineplot(data=median_df, x='nprops', y='AUC', linewidth=3, color='orange', label='Median AUC', marker='o', markersize=8)
sns.pointplot(data=median_df, x='nprops', y='AUC', color='orange')

# Adding labels and title
plt.title('Distribution of AUC Differences by Nprops Compared to Nprop=0', fontsize=18, color='white')
plt.xlabel('Nprops', fontsize=14, color='white')
plt.ylabel('AUC Difference from Nprop=0', fontsize=14, color='white')

# Customizing ticks and adding a horizontal line at 0
plt.xticks(range(5), fontsize=12, color='white')
plt.yticks(fontsize=12, color='white')
plt.axhline(0, color='white', linestyle='--', linewidth=1)
plt.grid(color='gray', linestyle='dashed', linewidth=0.5, alpha=0.5)

# only show from 0 to 4 (don't add any margin)
plt.xlim(0, 4)
plt.ylim(0.6,1.0)

plt.legend(fontsize=12)
plt.tight_layout()

# Assuming the save path matches your setup
plt.savefig('notebook/plots/auc_diff_by_nprops.png', facecolor='black')
