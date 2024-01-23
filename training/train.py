"""
This script prepares the data, runs the training, and saves the model.
"""

import argparse
import os
import sys
import pickle
import json
import logging
import pandas as pd
import time
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Change to CONF_FILE = "C:\Users\User\Desktop\M_Aghayan_MLE_EPAM8\settings.json" if you have problems with env variables
# CONF_FILE = os.getenv('CONF_PATH')
CONF_FILE = "C:/Users/User/Desktop/M_Aghayan_MLE_EPAM8/settings.json"
from utils import get_project_dir, configure_logging

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_file", help="Specify inference data file", default=conf['train']['table_name'])
parser.add_argument("--model_type", help="Specify the type of model to train",
                    choices=['decision_tree', 'random_forest', 'svm', 'logistic_regression'])
parser.add_argument("--model_path", help="Specify the path for the output model")


class DataProcessor():
    def __init__(self) -> None:
        pass

    def prepare_data(self, max_rows: int = None) -> pd.DataFrame:
        logging.info("Preparing data for training...")
        df = self.data_extraction(TRAIN_PATH)
        df = self.data_rand_sampling(df, max_rows)
        return df

    def data_extraction(self, path: str) -> pd.DataFrame:
        logging.info(f"Loading data from {path}...")
        return pd.read_csv(path)

    def data_rand_sampling(self, df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
        if not max_rows or max_rows < 0:
            logging.info('Max_rows not defined. Skipping sampling.')
        elif len(df) < max_rows:
            logging.info('Size of dataframe is less than max_rows. Skipping sampling.')
        else:
            df = df.sample(n=max_rows, replace=False, random_state=conf['general']['random_state'])
            logging.info(f'Random sampling performed. Sample size: {max_rows}')
        return df

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
    
class Training():
    def __init__(self, model_type=None) -> None:
        self.model_type = model_type or 'random_forest'
        self.model = self.initialize_model()

    def initialize_model(self):
        if self.model_type == 'decision_tree':
            return DecisionTreeClassifier(random_state=conf['general']['random_state'])
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(random_state=conf['general']['random_state'])
        elif self.model_type == 'svm':
            return SVC(random_state=conf['general']['random_state'])
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(random_state=conf['general']['random_state'])
        elif self.model_type == 'neural_network':
            input_size =  4
            hidden_size = 32
            output_size =  3
            return NeuralNetwork(input_size, hidden_size, output_size)
        else:
            raise ValueError(f"Invalid model_type: {self.model_type}. "
                             "Supported types are: 'decision_tree', 'random_forest', 'svm', 'logistic_regression', 'neural_network'.")

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        logging.info("Training the model...")
        
        if isinstance(self.model, nn.Module):  # Check if the model is a PyTorch model
            # Convert DataFrame to PyTorch tensor
            X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            for epoch in range(100):  # number of epochs
                optimizer.zero_grad()
                outputs = self.model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
            
        else:
            # Use scikit-learn model for other types
            self.model.fit(X_train, y_train)

    def run_training(self, df: pd.DataFrame, test_size: float = 0.33, model_path: str = None) -> None:
        logging.info("Running training...")
        X_train, X_test, y_train, y_test = self.data_split(df, test_size=test_size)
        start_time = time.time()
        self.train(X_train, y_train)
        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time} seconds.")
        self.test(X_test, y_test)
        self.save(model_path)

    def data_split(self, df: pd.DataFrame, test_size: float = 0.33) -> tuple:
        logging.info("Splitting data into training and test sets...")
        return train_test_split(df.drop(columns=['target']), df['target'], test_size=test_size,
                                random_state=conf['general']['random_state'])

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        logging.info("Training the model...")
        self.model.fit(X_train, y_train)

    def test(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
        logging.info("Testing the model...")
        y_pred = self.model.predict(X_test)
        res = f1_score(y_test, y_pred, average='weighted')
        logging.info(f"f1_score: {res}")
        return res

    def save(self, path: str) -> None:
        logging.info("Saving the model...")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        filename = f"{self.model_type}_model_{datetime.now().strftime(conf['general']['datetime_format'])}.pickle"
        if not path:
            path = os.path.join(MODEL_DIR, filename)
        else:
            path = os.path.join(MODEL_DIR, path)

        with open(path, 'wb') as f:
            pickle.dump(self.model, f)


def main():
    configure_logging()

    # Parse command line arguments
    args = parser.parse_args()

    data_proc = DataProcessor()
    tr = Training(model_type=args.model_type)

    df = data_proc.prepare_data(max_rows=conf['train']['data_sample'])
    tr.run_training(df, test_size=conf['train']['test_size'], model_path=args.model_path)


if __name__ == "__main__":
    main()