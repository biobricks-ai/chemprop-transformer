import matplotlib.pyplot as plt, pandas as pd, seaborn as sns

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

# AUC DIFFERENCE BY POSITION ============================================================
assays = df.groupby('assay').filter(lambda x: x['nprops'].nunique() == 10)['assay'].unique()
posdf = df[df['assay'].isin(assays)]
posdf.sort_values(by=['assay', 'nprops'], inplace=True)

# Correctly calculate the AUC difference compared to nprop = 0 for each assay
auc_at_zero = posdf[posdf['nprops'] == 0].set_index('assay')['AUC']
auc_at_zero = auc_at_zero.filter(lambda x: x["AUC"] < 0.8)
posdf['AUC_DIFF'] = posdf.apply(lambda row: row['AUC'] - auc_at_zero[row['assay']], axis=1)

# Filter out nprop = 0 since we're interested in the differences for >0
plotdf = posdf[['assay', 'nprops', 'AUC_DIFF']]

# Provided plotting code with minor adjustments for clarity
plt.style.use('dark_background')
plt.figure(figsize=(12, 6))

# Creating the violin plot
sns.violinplot(data=plotdf, x='nprops', y='AUC_DIFF', palette='coolwarm', inner='quartile')

# Adding labels and title
plt.title('Distribution of AUC Differences by Nprops Compared to Nprop=0', fontsize=18, color='white')
plt.xlabel('Nprops', fontsize=14, color='white')
plt.ylabel('AUC Difference from Nprop=0', fontsize=14, color='white')

# Customizing ticks and adding a horizontal line at 0
plt.xticks(fontsize=12, color='white')
plt.yticks(fontsize=12, color='white')
plt.axhline(0, color='white', linestyle='--', linewidth=1)

plt.grid(color='gray', linestyle='dashed', linewidth=0.5, alpha=0.5)
plt.tight_layout()

# Assuming the save path matches your setup
plt.savefig('notebook/plots/auc_diff_by_nprops.png', facecolor='black')


# SCRATCH ===============================================================================

assays = df.groupby('assay').filter(lambda x: x['nprops'].nunique() == 10)['assay'].unique()
posdf = df[df['assay'].isin(assays)]
iloss = posdf[posdf['nprops'] == 0].set_index('assay')['cross_entropy_loss']

# Calculate the improvement score
posdf['Improvement_Score'] = posdf.apply(lambda row: 100 * (iloss[row['assay']] - row['cross_entropy_loss']) / iloss[row['assay']], axis=1)

posdf.groupby('nprops').agg({'Improvement_Score': 'mean'})
plotdf = posdf[posdf['nprops'] > 0][['assay', 'nprops', 'cross_entropy_loss']]

# Plotting
plt.style.use('dark_background')
plt.figure(figsize=(12, 6))
sns.violinplot(data=plotdf, x='nprops', y='cross_entropy_loss', inner='quartile', cut=0)

# 'cut=0' limits the violin plot to the range of the observed data, potentially making it easier to focus on the main distribution without extreme tails.

# Adding labels and title
plt.title('Improvement Score by Nprops', fontsize=18, color='white')
plt.xlabel('Nprops', fontsize=14, color='white')
plt.ylabel('Improvement Score (%)', fontsize=14, color='white')

# Enhancing visibility
plt.xticks(fontsize=12, color='white')
plt.yticks(fontsize=12, color='white')
plt.axhline(0, color='white', linestyle='--', linewidth=1)
plt.grid(color='gray', linestyle='dashed', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig('notebook/plots/improvement_score_by_nprops.png', facecolor='black')
