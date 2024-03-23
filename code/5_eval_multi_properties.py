import numpy as np
import pandas as pd, tqdm, sklearn.metrics
import torch
import cvae.tokenizer, cvae.models.multitask_transformer as mt, cvae.utils
import importlib
importlib.reload(mt)

DEVICE = torch.device(f'cuda:0')
tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('data/processed/selfies_property_val_tokenizer')
model = mt.MultitaskTransformer.load("brick/mtransform2").to(DEVICE)

# EVALUATION LOOP ===================================================================
assay_indexes = torch.tensor(list(tokenizer.assay_indexes().values()), device=DEVICE)
value_indexes = torch.tensor(list(tokenizer.value_indexes().values()), device= DEVICE)

tst = mt.SequenceShiftDataset("data/tensordataset/multitask_tensors/tst", tokenizer)
tstdl = torch.utils.data.DataLoader(tst, batch_size=128, shuffle=False)
out_df = pd.DataFrame()

inp, teach, out = next(iter(tstdl))
for j in range(10):
    for i, (inp, teach, out) in tqdm.tqdm(enumerate(tstdl), total=len(tstdl)):
        inp, teach, out = inp.to(DEVICE), teach.to(DEVICE), out.to(DEVICE)
        
        # filter to instances with at least 10 properties
        x = torch.gt(torch.sum(torch.isin(out, value_indexes),dim=1),1)
        inp,teach, out = inp[x],teach[x],out[x]
        
        # get model predictions as a prob
        prob = torch.softmax(model(inp, teach),dim=2).detach()
        
        # get out assays and the assay with the highest prob
        assays = out[torch.isin(out, assay_indexes)].cpu().numpy()
        prob_assays = torch.argmax(prob, dim=2)[torch.isin(out, assay_indexes)].cpu().numpy()
        
        # get out values and the value with the highest prob and the prob of the `1`` value
        values = out[torch.isin(out, value_indexes)].cpu().numpy()
        probmax_vals = torch.argmax(prob, dim=2)[torch.isin(out, value_indexes)].cpu().numpy()
        rawprobs = prob[torch.isin(out, value_indexes)][:,value_indexes]
        probs = (rawprobs / rawprobs.sum(dim=1, keepdim=True))[:,1].cpu().numpy()
        
        # get position of each value in the out tensor
        # num_props = torch.sum(torch.isin(inp, assay_indexes), dim=1).cpu().numpy()
        num_props = torch.sum(torch.isin(out, assay_indexes), dim=1).cpu().numpy()
        position = torch.cat([torch.arange(size.item()) for size in num_props]).cpu().numpy()
        
        batch_df = pd.DataFrame({ 'batch': i, 'assay': assays, 'value': values, 'probs':probs, 'nprops':position })
        batch_df['prob_assays'] = prob_assays
        batch_df['prob_vals'] = probmax_vals
        
        out_df = pd.concat([out_df, batch_df], ignore_index=True)

sum(out_df['assay'] == out_df['prob_assays']) / len(out_df)
sum(out_df['value'] == out_df['prob_vals']) / len(out_df)

# GENERATE STRATIFIED EVALUATIONS FOR POSITION 0-9 ===============================

assay_metrics = []
grouped = out_df.groupby(['nprops','assay'])
# (position, assay), group = next(iter(grouped))
for (position,assay), group in tqdm.tqdm(grouped):
    y_true, y_pred = group['value'].values, group['probs'].values
    y_true = np.array([1 if x == 6097 else 0 for x in y_true])
    if sum(y_true==0) < 10 or sum(y_true==1) < 10 : continue
    auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
    acc = sklearn.metrics.accuracy_score(y_true, y_pred > 0.5)
    bac = sklearn.metrics.balanced_accuracy_score(y_true, y_pred > 0.5)
    assay_metrics.append({'nprops': position, 'assay': assay, 'AUC': auc, 'ACC': acc, 'BAC': bac, "NUM_POS": sum(y_true==1), "NUM_NEG": sum(y_true==0)})

metrics_df = pd.DataFrame(assay_metrics)
metrics_df.sort_values(by=['AUC'], inplace=True, ascending=False)
metrics_df.aggregate({'AUC': 'median', 'ACC': 'median', 'BAC': 'median'})
metrics_df
metrics_df.to_csv("metrics.csv")
metrics_df = pd.read_csv('metrics.csv')

position_df = metrics_df.groupby('nprops').aggregate({'AUC': 'median', 'ACC': 'median', 'BAC': 'median'})
# count nprops
position_df['count'] = metrics_df.groupby('nprops').size()
position_df.sort_values(by=['AUC'], inplace=True, ascending=False)
position_df
position_df.to_csv("position_metrics.csv")

auc, acc, bac = metrics_df['AUC'].median(), metrics_df['ACC'].median(), metrics_df['BAC'].median()
print(f"Metrics over position:\tAUC: {auc:.4f}\tACC: {acc:.4f}\tBAC: {bac:.4f}")

# ==========================================================
import pandas as pd

low_auc_threshold = 0.5  # Adjust this threshold as needed
low_auc_assays = metrics_df[metrics_df['AUC'] < low_auc_threshold]['assay'].unique()
low_auc_out_df = out_df[out_df['prob_assays'].isin(low_auc_assays)]
unique_low_auc_prob_assays = low_auc_out_df['prob_assays'].unique()
print(unique_low_auc_prob_assays)

#===========================================

from matplotlib import pyplot as plt

plt.style.use('dark_background')

# Create the figure and the histogram
plt.figure(figsize=(20, 10))
n, bins, patches = plt.hist(metrics_df['AUC'], bins=50, alpha=0.5, edgecolor='white', linewidth=1, color='turquoise')

# Add a line for the median AUC value
plt.axvline(auc, color='yellow', linestyle='dashed', linewidth=2)

# Annotate the median AUC value
median_annotation = f'Median AUC: {auc:.4f}'
plt.annotate(median_annotation, xy=(auc, max(n)), xytext=(auc, max(n) + max(n)*0.1),
            arrowprops=dict(facecolor='yellow', shrink=0.05),
            fontsize=18, color='yellow', fontweight='bold', ha='center')

# Label the bars with the count
for (rect, label) in zip(patches, n):
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height + 5, f'{int(label)}', ha='center', va='bottom', color='white', fontsize=12)

# Enhance titles and labels
plt.title('Histogram of AUC per Property', color='white', fontsize=26)
plt.xlabel('AUC', color='white', fontsize=22)
plt.ylabel('Number of Properties', color='white', fontsize=22)

# Improve tick marks
plt.xticks(fontsize=18, color='white')
plt.yticks(fontsize=18, color='white')

# Show grid
plt.grid(color='gray', linestyle='dashed', linewidth=0.5, alpha=0.5)

# Adjust the layout
plt.tight_layout()

# Save the plot to a file
plt.savefig('notebook/plots/multitask_transformer_metrics.png', facecolor='black')


#===========================================
# Filter metrics_df for position 9
position_9_df = metrics_df[metrics_df['position'] == 9]

# Calculate median AUC for position 9
auc_position_9 = position_9_df['AUC'].median()

# Create the histogram for position 9
plt.style.use('dark_background')
plt.figure(figsize=(20, 10))
n, bins, patches = plt.hist(position_9_df['AUC'], bins=20, alpha=0.5, edgecolor='white', linewidth=1.5, color='turquoise')

# Add a line for the median AUC value at position 9
plt.axvline(auc_position_9, color='yellow', linestyle='dashed', linewidth=2)

# Annotate the median AUC value
median_annotation = f'Median AUC at Position 9: {auc_position_9:.4f}'
plt.annotate(median_annotation, xy=(auc_position_9, max(n)), xytext=(auc_position_9, max(n) + max(n)*0.1),
             arrowprops=dict(facecolor='yellow', shrink=0.05),
             fontsize=18, color='yellow', fontweight='bold', ha='center')

# Label the bars with the count
for (rect, label) in zip(patches, n):
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height + 5, f'{int(label)}', ha='center', va='bottom', color='white', fontsize=12)

# Enhance titles and labels
plt.title('Histogram of AUC per Property at Position 9', color='white', fontsize=26)
plt.xlabel('AUC', color='white', fontsize=22)
plt.ylabel('Number of Properties', color='white', fontsize=22)

# Improve tick marks
plt.xticks(fontsize=18, color='white')
plt.yticks(fontsize=18, color='white')

# Show grid
plt.grid(color='gray', linestyle='dashed', linewidth=0.5, alpha=0.5)

# Adjust the layout
plt.tight_layout()

# Save the plot to a file
plt.savefig('notebook/plots/multitask_transformer_metrics_position_9.png', facecolor='black')

# Display the plot
plt.show()


#===========================================
import pandas as pd
import matplotlib.pyplot as plt

# Assuming metrics_df is the DataFrame you have already created
# Filter to include only positions from 0 to 9
filtered_df = metrics_df[metrics_df['position'].between(0, 9)]

# Group by position and calculate median AUC for each position
position_auc_df = filtered_df.groupby('position')['AUC'].median().reset_index()

# Create the line plot
plt.style.use('dark_background')
plt.figure(figsize=(12, 6))

# Plotting the line
plt.plot(position_auc_df['position'], position_auc_df['AUC'], color='turquoise', marker='o')

# Adding labels and title
plt.title('Median AUC Over Positions 0-9', fontsize=18, color='white')
plt.xlabel('Position', fontsize=14, color='white')
plt.ylabel('Median AUC', fontsize=14, color='white')

# Enhance the ticks and grid
plt.xticks(range(0, 10), fontsize=12, color='white')
plt.yticks(fontsize=12, color='white')
plt.grid(color='gray', linestyle='dashed', linewidth=0.5, alpha=0.5)

# Adjust the layout
plt.tight_layout()

# Save the plot
plt.savefig('notebook/plots/auc_over_positions.png', facecolor='black')

# Display the plot
plt.show()


#===========================================
import pandas as pd
import matplotlib.pyplot as plt

# Assuming metrics_df is the DataFrame you have already created
# Filter to include only positions from 0 to 9
filtered_df = metrics_df[metrics_df['position'].between(0, 9)]

# Pivot the DataFrame to get AUC values in columns per position
pivot_df = filtered_df.pivot(index='assay', columns='position', values='AUC')

# Calculate the difference in AUC between consecutive positions
auc_diff = pivot_df.diff(axis=1)

# Drop the first column (position 0) as it will be NaN after diff
auc_diff.drop(columns=[0], inplace=True)

# Calculate the median of these differences for each position transition
median_diffs = auc_diff.median()

# Create the line plot
plt.style.use('dark_background')
plt.figure(figsize=(12, 6))

# Plotting the line
plt.plot(median_diffs.index, median_diffs.values, color='turquoise', marker='o')

# Adding labels and title
plt.title('Median AUC Differences Over Positions 0-9', fontsize=18, color='white')
plt.xlabel('Position Transition', fontsize=14, color='white')
plt.ylabel('Median AUC Difference', fontsize=14, color='white')

# Enhance the ticks and grid
plt.xticks(range(1, 10), [f'{i}-{i+1}' for i in range(9)], fontsize=12, color='white')
plt.yticks(fontsize=12, color='white')
plt.grid(color='gray', linestyle='dashed', linewidth=0.5, alpha=0.5)

# Adjust the layout
plt.tight_layout()

# Save the plot
plt.savefig('notebook/plots/auc_diff_over_positions.png', facecolor='black')

# Display the plot
plt.show()
