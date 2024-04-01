import matplotlib.pyplot as plt, pandas as pd, seaborn as sns
import random

# SETUP =================================================================================
df = pd.read_csv('data/metrics/multitask_metrics.csv')
df.aggregate({'AUC': 'median', 'ACC': 'median', 'BAC': 'median', "cross_entropy_loss": 'median'})
# how many assays?
df['assay'].nunique()
auc = df['AUC'].median()

# AUC HISTOGRAM =========================================================================
def auc_histogram(df):
    assays = df.groupby('assay').filter(lambda x: x['nprops'].nunique() == 5)['assay'].unique()
    plotdf = df[df['nprops'].isin([0, 1, 2, 3, 4]) & df['assay'].isin(assays)]

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
    plt.savefig('notebook/plots/multitask_transformer_metrics.png', facecolor='none', transparent=True)

auc_histogram(df)

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