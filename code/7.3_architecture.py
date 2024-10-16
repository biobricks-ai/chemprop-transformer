import matplotlib.pyplot as plt
import numpy as np
import biobricks as bb
import pandas as pd
from matplotlib.patches import Rectangle, Arrow

ch = bb.assets('chemharmony')

activities = pd.read_parquet(ch.activities_parquet)
# group by smiles and find the smiles with the most assays
smiles_counts = activities.groupby('smiles').size().reset_index(name='count')
smiles_counts = smiles_counts.sort_values(by='count', ascending=False)

benzene = activities[activities['smiles'] == 'c1ccccc1']

properties = pd.read_parquet(ch.properties_parquet)
proptitles = pd.read_parquet(ch.property_titles_parquet)

df = benzene \
    .merge(properties, on=['source', 'pid'], how='inner') \
    .merge(proptitles, on='pid', how='inner') \
    [['inchi','pid','source','title','value']]

# give me 3 examples with the property `data` for each `source` 
test = df.drop_duplicates(subset='pid').groupby('source').head(3)
test = test[['inchi', 'source', 'title', 'value']].sort_values(by='source').reset_index(drop=True)
test
test.iloc[0]['value'] == test.iloc[1]['value']

c = test.iloc[2]['pid']

print(a == b)
print(a == c)
print(b == c)

# check if row 0 and 1 are the same
test.iloc[1] == test.iloc[2]
