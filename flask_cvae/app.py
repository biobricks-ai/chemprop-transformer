from flask import Flask, request, jsonify
from rdkit import Chem
from rdkit.Chem import Draw

import flask_cvae.datastore as datastore
import numpy as np
import logging, time, json, torch

app = Flask(__name__)
logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s')

logging.info("loading analog finder")
start_time = time.time()  # Start the timer
analogue_finder = datastore.AnalogueFinder.load("analoguefinder.pkl")
end_time = time.time()  # End the timer
logging.info(f"analogue finder loaded in {end_time - start_time:.2f} seconds")

@app.route('/predict', methods=['GET'])
def predict():
    inchi = request.args.get('inchi')
    label = request.args.get('label')
    k = request.args.get('k')
    mol = None
    
    try:
        mol = Chem.MolFromInchi(inchi)
        label = int(label)
        k = int(k)
        
    except (ValueError, TypeError):
        return "Invalid arguments: 'label' and 'k' must be integers, 'inchi' must be a string that can be parsed into a molecule", 400

    try:
        smiles = Chem.MolToSmiles(mol)
        res = vars(analogue_finder.knn("vae", smiles, label=label, k=5))
        jsres = {k: v.tolist() if isinstance(v, (np.ndarray, torch.Tensor)) else v for k, v in res.items()}

        return json.dumps(jsres)

    except Exception as e:
        logging.error(f"Exception occurred: {str(e)}")
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run()
