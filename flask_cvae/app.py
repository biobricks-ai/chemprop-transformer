from flask import Flask, request, jsonify
from rdkit import Chem
from rdkit.Chem import Draw

import flask_cvae.datastore as datastore

app = Flask(__name__)

analogue_finder = datastore.AnalogueFinder.load("analoguefinder.pkl")

@app.route('/predict', methods=['GET'])
def predict():
    inchi = request.args.get('inchi')
    
    if inchi is None:
        return jsonify(error="No InChI provided"), 400

    try:
        parsed = Chem.MolFromInchi(inchi)
        if parsed is None:
            return jsonify(error="Invalid InChI provided"), 400

        smiles = Chem.MolToSmiles(parsed)
        res = analogue_finder.knn("vae", smiles, label=1, k=5)
        return jsonify(prediction=0.5)

    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run()
