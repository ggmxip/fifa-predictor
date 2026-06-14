"""Ensemble model combining ML and statistical predictions."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import joblib
from pathlib import Path


class SKLearnWrapper:
    """Wraps a non-sklearn model (Poisson, Elo) to .fit(X,y) / .predict_proba(X) interface."""

    def __init__(self, model, team_pairs_map):
        self.model = model
        self.team_pairs_map = team_pairs_map

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        n = len(X)
        probs = np.full((n, 3), 1.0 / 3)
        for i in range(n):
            idx = X.index[i] if hasattr(X, 'index') else i
            pair = self.team_pairs_map.get(idx, ('Unknown', 'Unknown'))
            t1, t2 = pair
            try:
                probs[i] = self.model.predict_match_probs(t1, t2)
            except Exception:
                pass
        return probs


class HybridEnsemble:
    """Stacked ensemble combining ML classifiers and statistical models."""

    def __init__(self, config):
        self.config = config
        self.base_models_ = {}
        self.meta_learner_ = None
        self.is_fitted_ = False
        self.model_weights_ = None

    def add_model(self, name, model):
        """Register a base model."""
        self.base_models_[name] = model

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the ensemble using stacking."""
        cv = StratifiedKFold(n_splits=self.config['ensemble']['cv_folds'], shuffle=True, random_state=42)

        model_names = list(self.base_models_.keys())
        n_models = len(model_names)
        n_samples = len(X_train)

        ml_names = [n for n in model_names if not isinstance(self.base_models_[n], SKLearnWrapper)]
        stat_names = [n for n in model_names if isinstance(self.base_models_[n], SKLearnWrapper)]

        meta_features = np.zeros((n_samples, n_models * 3))

        for i, name in enumerate(ml_names):
            model = self.base_models_[name]
            preds = np.zeros((n_samples, 3))
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_tr, X_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
                model.fit(X_tr, y_tr)
                preds[val_idx] = model.predict_proba(X_cv)
            meta_features[:, self._col_idx(name, model_names)*3:(self._col_idx(name, model_names)+1)*3] = preds

        for name in stat_names:
            model = self.base_models_[name]
            preds = model.predict_proba(X_train)
            idx = self._col_idx(name, model_names)
            meta_features[:, idx*3:(idx+1)*3] = preds

        self.meta_learner_ = LogisticRegression(
            solver='lbfgs', max_iter=1000, random_state=42
        )
        self.meta_learner_.fit(meta_features, y_train)

        if self.config['ensemble']['use_calibration']:
            self.meta_learner_ = CalibratedClassifierCV(
                self.meta_learner_, cv=3, method='sigmoid'
            )
            self.meta_learner_.fit(meta_features, y_train)

        self._compute_weights(X_val if X_val is not None else X_train,
                              y_val if y_val is not None else y_train)

        self.is_fitted_ = True
        return self

    def _col_idx(self, name, model_names):
        return list(model_names).index(name)

    def _compute_weights(self, X, y):
        """Compute optimal weights based on validation performance."""
        model_names = list(self.base_models_.keys())
        losses = []
        for name in model_names:
            model = self.base_models_[name]
            probs = model.predict_proba(X)
            losses.append(log_loss(y, probs))

        inv_losses = np.array([1.0 / (l + 1e-8) for l in losses])
        self.model_weights_ = inv_losses / inv_losses.sum()

    def predict_proba(self, X):
        """Predict with calibrated ensemble."""
        meta_features = self._get_meta_features(X)
        return self.meta_learner_.predict_proba(meta_features)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_weighted(self, X):
        """Weighted average of base model probabilities (bypasses meta-learner)."""
        model_names = list(self.base_models_.keys())
        avg_probs = np.zeros((len(X), 3))
        for i, name in enumerate(model_names):
            probs = self.base_models_[name].predict_proba(X)
            avg_probs += self.model_weights_[i] * probs
        return avg_probs / avg_probs.sum(axis=1, keepdims=True)

    def _get_meta_features(self, X):
        model_names = list(self.base_models_.keys())
        n_models = len(model_names)
        meta_features = np.zeros((len(X), n_models * 3))
        for i, name in enumerate(model_names):
            probs = self.base_models_[name].predict_proba(X)
            meta_features[:, i*3:(i+1)*3] = probs
        return meta_features

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
