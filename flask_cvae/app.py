from flask import Flask, request, jsonify
from werkzeug.serving import WSGIRequestHandler
import sqlite3
import threading
import logging

import sys
sys.path.append('./flask_cvae')
from flask_cvae.predictor import Predictor, Prediction

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s',
                    handlers=[logging.StreamHandler()])


WSGIRequestHandler.protocol_version = "HTTP/1.1"

app = Flask(__name__)
app.config['TIMEOUT'] = 300  # 5 minutes in seconds

predict_lock = threading.Lock()
predictor = Predictor()

@app.route('/predict_all', methods=['GET'])
def predict_all():
    logging.info(f"Predicting all properties for inchi: {request.args.get('inchi')}")
    inchi = request.args.get('inchi')
    with predict_lock:
        property_predictions : list[Prediction] = predictor.predict_all_properties(inchi)

    return jsonify(property_predictions)

@app.route('/predict', methods=['GET'])
def predict():
    logging.info(f"Predicting property for inchi: {request.args.get('inchi')} and property token: {request.args.get('property_token')}")
    inchi = request.args.get('inchi')
    property_token = request.args.get('property_token', None)
    if inchi is None or property_token is None:
        return jsonify({'error': 'inchi and property token parameters are required'})
    
    with predict_lock:
        mean_value = float(predictor.predict_property(inchi, int(property_token)))

    return jsonify({"inchi": inchi, "property_token": property_token, "positive_prediction": mean_value})

@app.route('/health', methods=['GET'])
def health_check():
    """Quick health check that returns immediately."""
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    app.run(debug=True)
