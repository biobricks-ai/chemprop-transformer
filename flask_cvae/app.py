from flask import Flask, request, jsonify
import sqlite3
import threading
import logging
from .predictor import Predictor, Prediction

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s',
                    handlers=[logging.StreamHandler()])

predict_lock = threading.Lock()
cvaesql = sqlite3.connect('brick/cvae.sqlite')
cvaesql.row_factory = sqlite3.Row  # This enables column access by name

psqlite = sqlite3.connect('flask_cvae/predictions.sqlite')
cmd = "CREATE TABLE IF NOT EXISTS prediction (inchi TEXT, property_token INTEGER, value float)"
psqlite.execute(cmd)
cmd = "CREATE INDEX IF NOT EXISTS idx_inchi_property_token ON prediction (inchi, property_token)"
psqlite.execute(cmd)

app = Flask(__name__)
predictor = Predictor(psqlite)

@app.route('/predict', methods=['GET'])
def predict():
    logging.info(f"Predicting property for inchi: {request.args.get('inchi')} and property token: {request.args.get('property_token')}")
    inchi = request.args.get('inchi')
    property_token = request.args.get('property_token', None)
    if inchi is None or property_token is None:
        return jsonify({'error': 'inchi and property token parameters are required'})
    
    with predict_lock:
        mean_value = float(predictor.cached_predict_property(inchi, int(property_token)))

    return jsonify({"inchi": inchi, "property_token": property_token, "positive_prediction": mean_value})

if __name__ == '__main__':
    app.run(debug=True)
