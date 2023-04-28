import biobricks as bb
import dvc.api
import os


os.makedirs('data/', exist_ok=True)

bb.install('chemharmony')
chemharmony = bb.load('chemharmony')

activities = chemharmony.activities.read().to_pandas()

activities['group'] = activities.groupby(['pid', 'smiles']).ngroup()
activities = activities.drop_duplicates(subset=['group'])
activities = activities[['smiles', 'pid', 'binary_value']]
activities.columns = ['smiles','assay','value']

activities.to_csv('data/RawChemHarmony.csv')