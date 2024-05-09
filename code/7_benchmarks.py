import pandas as pd, sqlite3, seaborn as sns, matplotlib.pyplot as plt

#%% TOXCAST BENCHMARK ===========================================================
conn = sqlite3.connect('brick/cvae.sqlite')

# get all the property_tokens for tox21 properties
prop_src = pd.read_sql("SELECT property_token,title,source FROM property p INNER JOIN source s on p.source_id = s.source_id", conn)

# pull in multitask_metrics
evaldf = pd.read_csv('data/metrics/multitask_metrics.csv')\
    .merge(prop_src, left_on='assay', right_on='property_token', how='inner')

# get the median AUC for each property
evaldf.aggregate({'AUC': 'median','cross_entropy_loss':'median','assay':'count'}) # 89% median auc, .482 median cross entropy loss
evaldf.groupby(['source','nprops']).aggregate({'AUC': 'median','assay':'count'}).sort_values(by='AUC',ascending=False)
evaldf[evaldf['source'] == 'tox21'].groupby(['nprops']).aggregate({'AUC': 'median','assay':'count'}).sort_values(by='AUC',ascending=False)
#              AUC  assay
# nprops                 
# 2       0.909173    110
# 3       0.902808    110
# 4       0.900132    110
# 1       0.871136     66
# 0       0.829932     18

#%% CATEGORICAL BENCHMARK ======================================================

pcat = pd.read_sql("""SELECT property_token,title,category,strength FROM property p 
                       INNER JOIN property_category pc ON p.property_id = pc.property_id
                       INNER JOIN category c on pc.category_id = c.category_id""", 
                       conn)

titles = pcat[pcat['category'] == 'acute inhalation toxicity']
titles = titles[titles['strength'] > 8.0]

# print titles
for i, row in titles[['title']].iterrows():
    print(row['title'])
    
evalcat = pd.read_csv('data/metrics/multitask_metrics.csv')\
    .merge(pcat, left_on='assay', right_on='property_token', how='inner')\
    .query('strength > 8.0')

# get the median AUC for each property
evalcat.aggregate({'AUC': 'median','assay':'count'})
pdf = evalcat.groupby(['category','nprops']).aggregate({'AUC': 'median','assay':'count'}).reset_index().sort_values(by=['category','AUC'],ascending=False)

# create heatmap of AUC's with range 0.5 to 1.0 red to green 
# rows should be categories, columns should be nprops
heatmap_data = pdf.pivot(index="category", columns="nprops", values="AUC")
heatmap_data.index = heatmap_data.index.map(lambda x: x[:25] if len(x) > 25 else x)


# Create the heatmap using seaborn
cmap = sns.blend_palette(["#2a2a2a", "#34eb37"], as_cmap=True)

# Create the heatmap using seaborn
plt.figure(figsize=(12, 9))
ax = sns.heatmap(heatmap_data, cmap=cmap, annot=False, square=True, fmt=".2f",
                 linewidths=0.5, linecolor='black', cbar_kws={"shrink": .8})

plt.title('Heatmap of Median AUC by Category and Number of Properties', color='white', fontsize=20)
plt.xlabel('Number of Properties', fontsize=15, color='white')
plt.ylabel('Category', fontsize=15, color='white')

# Setting the tick labels color for row and column names
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# Improve the appearance of the colorbar
cbar = ax.collections[0].colorbar
cbar.set_label('Median AUC', color='white')
cbar.ax.yaxis.set_tick_params(color='white')
cbar.ax.yaxis.set_ticklabels([f'{x:.2f}' for x in cbar.get_ticks()], color='white')

# Save the figure to a file
plt.savefig('notebook/plots/category_auc_heatmap.png',facecolor='black',bbox_inches='tight',dpi=300)