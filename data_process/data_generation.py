# Importing required libraries
import pandas as pd
import logging
import os
import sys
import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
from utils import singleton, get_project_dir, configure_logging

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = os.getenv('CONF_PATH')

# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
logger.info("Defining paths...")
DATA_DIR = get_project_dir(conf['general']['data_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])
INFERENCE_PATH = os.path.join(DATA_DIR, conf['inference']['inp_table_name'])

# Singleton class for processing Iris dataset
@singleton
class IrisDataProcessor():
    def __init__(self):
        self.df = None

    # Method to load Iris dataset
    def load_iris_data(self):
        iris = load_iris()
        iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        iris_data['target'] = iris.target
        self.df = iris_data

    # Method to split and save data
    def process_and_split(self, train_file_path, inference_file_path):
        logger.info("Splitting Iris dataset...")
        train_data, inference_data = train_test_split(self.df, test_size=0.2, random_state=42)
        train_data.to_csv(train_file_path, index=False)
        inference_data.to_csv(inference_file_path, index=False)

# Main execution
if __name__ == "__main__":
    configure_logging()
    logger.info("Starting script...")
    
    processor = IrisDataProcessor()
    processor.load_iris_data()
    processor.process_and_split(TRAIN_PATH, INFERENCE_PATH)
    
    logger.info("Script completed successfully.")