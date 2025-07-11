from flask import Flask, jsonify, request
from loguru import logger
import pandas as pd

from src.inference import load_pipline, predict
from config.variables import PIPLINE_PATH

app = Flask(__name__)

logger.info("Starting the Flask application...")
logger.info("Loading the pipline...")
PIPLINE = load_pipline(PIPLINE_PATH)
logger.info("Model loaded successfully.")

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def prediction():
    try:
        logger.info('Recived a prediction request')
        data = request.json
        logger.info('Data recived for prediction')

        df = pd.DataFrame(data, index=[0])
        df = df.reset_index(drop=True)
        logger.info('Data converted to DataFrame')
        logger.debug(f"DataFrame: {df.shape}")

        prediction = predict(PIPLINE, df)
        logger.info('Prediction made successfully.')

        classes = ["Positive", "Negative"]
        prediction = classes[prediction[0]]
        logger.debug(f"Prediction: {prediction}")

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

    return jsonify(({"prediction": prediction}))
