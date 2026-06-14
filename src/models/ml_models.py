"""ML models for match outcome prediction."""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout=0.3, num_classes=3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TorchClassifier:
    """Wrapper for PyTorch neural network compatible with sklearn interface."""

    def __init__(self, hidden_layers=None, dropout=0.3, learning_rate=0.001,
                 batch_size=64, epochs=200, patience=20, random_state=42):
        self.hidden_layers = hidden_layers or [128, 64, 32]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.model_ = None
        self.scaler_ = StandardScaler()
        self.input_dim_ = None
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        self.input_dim_ = X.shape[1]
        X_scaled = self.scaler_.fit_transform(X)

        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.long)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_ = NeuralNet(self.input_dim_, self.hidden_layers, self.dropout, len(self.classes_))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model_.train()
            epoch_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        return self

    def predict_proba(self, X):
        X_scaled = self.scaler_.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_t)
            probs = torch.softmax(outputs, dim=1).numpy()
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class MLModelFactory:
    """Factory for creating and training ML models."""

    def __init__(self, config):
        self.config = config
        self.models_dir = Path('models')
        self.models_dir.mkdir(exist_ok=True)

    def create_xgboost(self):
        params = self.config['models']['xgboost']
        model = XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            gamma=params['gamma'],
            random_state=params['random_state'],
            eval_metric='mlogloss',
            objective='multi:softprob',
            num_class=3,
        )
        return model

    def create_lightgbm(self):
        params = self.config['models']['lightgbm']
        return LGBMClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            num_leaves=params['num_leaves'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            random_state=params['random_state'],
            objective='multiclass',
            num_class=3,
            metric='multi_logloss',
            verbose=-1,
        )

    def create_random_forest(self):
        params = self.config['models']['random_forest']
        return RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            random_state=params['random_state'],
            n_jobs=-1,
        )

    def create_neural_network(self):
        params = self.config['models']['neural_network']
        return TorchClassifier(
            hidden_layers=params['hidden_layers'],
            dropout=params['dropout'],
            learning_rate=params['learning_rate'],
            batch_size=params['batch_size'],
            epochs=params['epochs'],
            patience=params['patience'],
        )

    def train_model(self, model, X_train, y_train, X_val=None, y_val=None, model_name='model'):
        """Train a model with optional early stopping."""
        if hasattr(model, 'fit'):
            if X_val is not None and hasattr(model, 'early_stopping_rounds'):
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
            else:
                model.fit(X_train, y_train)
        return model

    def save_model(self, model, name):
        path = self.models_dir / f'{name}.joblib'
        joblib.dump(model, path)
        return path

    def load_model(self, name):
        path = self.models_dir / f'{name}.joblib'
        return joblib.load(path) if path.exists() else None
