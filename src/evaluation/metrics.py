"""Evaluation metrics and visualization for model performance."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, log_loss, brier_score_loss,
                             roc_auc_score, confusion_matrix, classification_report)
from pathlib import Path


class Evaluator:
    """Evaluates model predictions against ground truth."""

    def __init__(self, config):
        self.config = config
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        self.results_ = []

    def evaluate(self, y_true, y_pred, y_proba, model_name='model', tournament=''):
        """Compute all metrics for a set of predictions."""
        result = {
            'model': model_name,
            'tournament': tournament,
            'n_matches': len(y_true),
            'accuracy': accuracy_score(y_true, y_pred),
            'log_loss': log_loss(y_true, y_proba),
        }

        try:
            result['roc_auc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
        except Exception:
            result['roc_auc_ovr'] = None

        brier = 0
        for c in range(3):
            y_bin = (y_true == c).astype(int)
            brier += brier_score_loss(y_bin, y_proba[:, c])
        result['brier_score'] = brier / 3

        cm = confusion_matrix(y_true, y_pred)
        result['confusion_matrix'] = cm

        result['classification_report'] = classification_report(
            y_true, y_pred, target_names=['Home Win', 'Draw', 'Away Win'], output_dict=True
        )

        self.results_.append(result)
        return result

    def print_report(self, result):
        """Print a formatted evaluation report."""
        print(f"\n{'='*55}")
        print(f"  Model: {result['model']} | Tournament: {result['tournament']}")
        print(f"{'='*55}")
        print(f"  Matches:      {result['n_matches']}")
        print(f"  Accuracy:     {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
        print(f"  Log Loss:     {result['log_loss']:.4f}")
        print(f"  Brier Score:  {result['brier_score']:.4f}")
        print(f"  ROC-AUC (OVR): {result.get('roc_auc_ovr', 'N/A')}")
        print(f"\n  Confusion Matrix:")
        print(f"                 Pred H    Pred D    Pred A")
        cm = result['confusion_matrix']
        labels = ['True H', 'True D', 'True A']
        for i, label in enumerate(labels):
            print(f"  {label:<10} {cm[i,0]:<9} {cm[i,1]:<9} {cm[i,2]:<9}")
        print(f"{'='*55}\n")

    def plot_confusion_matrix(self, result, save_path=None):
        """Plot confusion matrix heatmap."""
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Home Win', 'Draw', 'Away Win'],
                    yticklabels=['Home Win', 'Draw', 'Away Win'], ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f"{result['model']} — {result['tournament']}")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def plot_calibration(self, y_true, y_proba, model_name='', save_path=None):
        """Plot calibration curve."""
        from sklearn.calibration import calibration_curve
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        class_names = ['Home Win', 'Draw', 'Away Win']

        for c in range(3):
            y_bin = (y_true == c).astype(int)
            prob_pred, prob_true = calibration_curve(y_bin, y_proba[:, c], n_bins=10)
            axes[c].plot(prob_pred, prob_true, marker='o', linewidth=1, label=f'Class {c}')
            axes[c].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
            axes[c].set_xlabel('Mean Predicted Probability')
            axes[c].set_ylabel('Fraction of Positives')
            axes[c].set_title(class_names[c])
            axes[c].legend()

        plt.suptitle(f'Calibration Curves — {model_name}', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def summary_report(self, save_path=None):
        """Print comparison of all evaluated models."""
        df = pd.DataFrame(self.results_)
        cols = ['model', 'tournament', 'n_matches', 'accuracy', 'log_loss', 'brier_score']
        print(f"\n{'='*70}")
        print(f"  SUMMARY — Model Comparison")
        print(f"{'='*70}")
        print(df[cols].to_string(index=False))
        print(f"{'='*70}")

        if save_path:
            df[cols].to_csv(save_path, index=False)
        return df
