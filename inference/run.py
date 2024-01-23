"""
Script loads the latest trained model, data for inference and predicts results.
Imports necessary packages and modules.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from typing import List

import pandas as pd
import torch
from torch import nn
from sklearn.metrics import f1_score

# Adds the root directory to the system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = os.getenv('CONF_PATH')

from utils import get_project_dir, configure_logging

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
RESULTS_DIR = get_project_dir(conf['general']['results_dir'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--infer_file",
                    help="Specify inference data file",
                    default=conf['inference']['inp_table_name'])
parser.add_argument("--out_path",
                    help="Specify the path to the output table")


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class Inference():
    def __init__(self, model_type=None):
        self.model_type = model_type or 'random_forest'
        self.model = self.load_model()

    def load_model(self):
        model_path = get_latest_model_path()
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model

    def preprocess_data(self, df):
        # Implement any necessary data preprocessing steps here
        # For example, scaling or encoding categorical variables
        return df

    def run_inference(self, df):
        logging.info("Running inference...")
        df = self.preprocess_data(df)

        if isinstance(self.model, NeuralNetwork):
            # Convert DataFrame to PyTorch tensor
            X_tensor = torch.tensor(df.values, dtype=torch.float32)
            outputs = self.model(X_tensor)
            _, predictions = torch.max(outputs, 1)
            return predictions.numpy()
        else:
            return self.model.predict(df)

    def evaluate_results(self, y_true, y_pred):
        logging.info("Evaluating results...")
        res = f1_score(y_true, y_pred, average='weighted')
        logging.info(f"f1_score: {res}")
        return res


def get_latest_model_path(model_dir: str, datetime_format: str) -> str:
    """Gets the path of the latest saved model"""
    latest = None
    for (dirpath, dirnames, filenames) in os.walk(model_dir):
        for filename in filenames:
            # Extract only the datetime part from the filename
            datetime_part = ''.join(c for c in filename if c.isdigit() or c in {'_', '.'})
            if not latest or datetime.strptime(datetime_part, datetime_format) < \
                    datetime.strptime(latest, datetime_format):
                latest = datetime_part
    return os.path.join(model_dir, f'model_{latest}.pickle')



def get_model_by_path(path: str):
    """Loads and returns the specified model"""
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
            logging.info(f'Path of the model: {path}')
            return model
    except Exception as e:
        logging.error(f'An error occurred while loading the model: {e}')
        sys.exit(1)


def get_inference_data(path: str) -> pd.DataFrame:
    """loads and returns data for inference from the specified csv file"""
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.error(f"An error occurred while loading inference data: {e}")
        sys.exit(1)


def store_results(results: pd.DataFrame, path: str = None) -> None:
    """Store the prediction results in 'results' directory with the current datetime as a filename"""
    if not path:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        path = datetime.now().strftime(conf['general']['datetime_format']) + '.csv'
        path = os.path.join(RESULTS_DIR, path)
    pd.DataFrame(results).to_csv(path, index=False)
    logging.info(f'Results saved to {path}')


def main():
    """Main function"""
    configure_logging()
    args = parser.parse_args()

    latest_model_path = get_latest_model_path(MODEL_DIR, conf['general']['datetime_format'])

    inference_processor = Inference(model_type=args.model_type, model_path=latest_model_path)
    infer_file = args.infer_file
    infer_data = get_inference_data(os.path.join(DATA_DIR, infer_file))
    predictions = inference_processor.run_inference(infer_data)
    store_results(predictions, args.out_path)

    logging.info(f'Prediction results: {predictions}')


if __name__ == "__main__":
    main()
