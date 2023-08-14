import dvc.api, os, biobricks as bb, pathlib

chemharmony = bb.load('chemharmony').activities

activities = chemharmony.read().to_pandas()[['smiles','pid','binary_value']]
activities = activities.rename(columns={'pid':'assay','binary_value':'value'})

params = dvc.api.params_show()
outFolder = pathlib.Path("data/raw")
os.makedirs(outFolder, exist_ok=True, parents=True)

activities.to_csv('{}/RawChemHarmony.csv'.format(outFolder))