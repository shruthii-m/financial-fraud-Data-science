"""
Model Evaluation Module for Financial Fraud Detection

This module contains comprehensive evaluation metrics and visualization
functions for assessing fraud detection model performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score, log_loss
)
from sklearn.calibration import calibration_curve
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import logging
from typing import Dict, List, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    A comprehensive class for evaluating fraud detection models.
    """
    
    def __init__(self):
        self.evaluation_results = {}
        
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              y_prob: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate basic classification metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_prob (np.ndarray): Predicted probabilities (optional)
            
        Returns:
            Dict[str, float]: Dictionary of basic metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary'),
            'specificity': self._calculate_specificity(y_true, y_pred)
        }
        
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
            metrics['log_loss'] = log_loss(y_true, y_prob)
        
        return metrics
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate specificity (True Negative Rate).
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            
        Returns:
            float: Specificity score
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def calculate_business_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 transaction_amounts: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate business-specific metrics for fraud detection.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            transaction_amounts (np.ndarray): Transaction amounts (optional)
            
        Returns:
            Dict[str, float]: Dictionary of business metrics
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = {
            'fraud_detection_rate': tp / (tp + fn) if (tp + fn) > 0 else 0.0,  # Same as recall
            'false_alarm_rate': fp / (fp + tn) if (fp + tn) > 0 else 0.0,  # Same as 1 - specificity
            'precision_fraud': tp / (tp + fp) if (tp + fp) > 0 else 0.0,  # Precision for fraud class
        }
        
        if transaction_amounts is not None:
            # Calculate financial impact metrics
            fraud_mask = y_true == 1
            detected_fraud_mask = (y_true == 1) & (y_pred == 1)
            
            total_fraud_amount = np.sum(transaction_amounts[fraud_mask])
            detected_fraud_amount = np.sum(transaction_amounts[detected_fraud_mask])
            
            metrics['fraud_amount_detected_rate'] = (
                detected_fraud_amount / total_fraud_amount if total_fraud_amount > 0 else 0.0
            )
            metrics['total_fraud_amount'] = total_fraud_amount
            metrics['detected_fraud_amount'] = detected_fraud_amount
            metrics['saved_amount'] = detected_fraud_amount
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            labels: List[str] = None, normalize: bool = False) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            labels (List[str]): Class labels
            normalize (bool): Whether to normalize the matrix
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        if labels is None:
            labels = ['Legitimate', 'Fraud']
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
        else:
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title('Confusion Matrix')
        
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, 
                      model_name: str = "Model") -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true (np.ndarray): True labels
            y_prob (np.ndarray): Predicted probabilities
            model_name (str): Name of the model
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                                   model_name: str = "Model") -> plt.Figure:
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true (np.ndarray): True labels
            y_prob (np.ndarray): Predicted probabilities
            model_name (str): Name of the model
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap_score = average_precision_score(y_true, y_prob)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, label=f'{model_name} (AP = {ap_score:.3f})', linewidth=2)
        
        # Add baseline (random classifier)
        baseline = np.sum(y_true) / len(y_true)
        ax.axhline(y=baseline, color='k', linestyle='--', 
                  label=f'Random Classifier (AP = {baseline:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_feature_importance(self, model: Any, feature_names: List[str], 
                              top_n: int = 20) -> plt.Figure:
        """
        Plot feature importance for tree-based models.
        
        Args:
            model (Any): Trained model with feature_importances_ attribute
            feature_names (List[str]): Names of the features
            top_n (int): Number of top features to display
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature importance attribute")
            return None
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(top_n), importances[indices])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Feature Importances')
        ax.invert_yaxis()
        
        return fig
    
    def plot_prediction_distribution(self, y_prob: np.ndarray, y_true: np.ndarray) -> plt.Figure:
        """
        Plot distribution of prediction probabilities by class.
        
        Args:
            y_prob (np.ndarray): Predicted probabilities
            y_true (np.ndarray): True labels
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot distributions
        legitimate_probs = y_prob[y_true == 0]
        fraud_probs = y_prob[y_true == 1]
        
        ax.hist(legitimate_probs, bins=50, alpha=0.7, label='Legitimate', 
               color='blue', density=True)
        ax.hist(fraud_probs, bins=50, alpha=0.7, label='Fraud', 
               color='red', density=True)
        
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Predicted Probabilities by Class')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                             n_bins: int = 10, model_name: str = "Model") -> plt.Figure:
        """
        Plot calibration curve to assess probability calibration.
        
        Args:
            y_true (np.ndarray): True labels
            y_prob (np.ndarray): Predicted probabilities
            n_bins (int): Number of bins for calibration curve
            model_name (str): Name of the model
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins
        )
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", 
               label=f'{model_name}', linewidth=2, markersize=8)
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax.set_ylabel('Fraction of Positives')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_title('Calibration Curve (Reliability Diagram)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def create_interactive_dashboard(self, y_true: np.ndarray, y_prob: np.ndarray,
                                   feature_names: List[str] = None, 
                                   model: Any = None) -> go.Figure:
        """
        Create an interactive evaluation dashboard using Plotly.
        
        Args:
            y_true (np.ndarray): True labels
            y_prob (np.ndarray): Predicted probabilities
            feature_names (List[str]): Feature names for importance plot
            model (Any): Trained model for feature importance
            
        Returns:
            go.Figure: Plotly figure with subplots
        """
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        
        # Calculate Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ROC Curve', 'Precision-Recall Curve', 
                          'Prediction Distribution', 'Feature Importance'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # ROC Curve
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC = {roc_auc:.3f})',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], name='Random',
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # Precision-Recall Curve
        fig.add_trace(
            go.Scatter(x=recall, y=precision, name=f'PR (AUC = {pr_auc:.3f})',
                      line=dict(color='green', width=2)),
            row=1, col=2
        )
        
        # Prediction Distribution
        legitimate_probs = y_prob[y_true == 0]
        fraud_probs = y_prob[y_true == 1]
        
        fig.add_trace(
            go.Histogram(x=legitimate_probs, name='Legitimate', opacity=0.7,
                        nbinsx=30, histnorm='probability density'),
            row=2, col=1
        )
        fig.add_trace(
            go.Histogram(x=fraud_probs, name='Fraud', opacity=0.7,
                        nbinsx=30, histnorm='probability density'),
            row=2, col=1
        )
        
        # Feature Importance (if model and feature names are provided)
        if model is not None and feature_names is not None and hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]  # Top 15 features
            
            fig.add_trace(
                go.Bar(x=importances[indices], 
                      y=[feature_names[i] for i in indices],
                      orientation='h', name='Importance'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Model Evaluation Dashboard"
        )
        
        return fig
    
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 y_prob: np.ndarray = None, 
                                 transaction_amounts: np.ndarray = None,
                                 model_name: str = "Model") -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_prob (np.ndarray): Predicted probabilities
            transaction_amounts (np.ndarray): Transaction amounts
            model_name (str): Name of the model
            
        Returns:
            Dict[str, Any]: Comprehensive evaluation report
        """
        logger.info(f"Generating evaluation report for {model_name}")
        
        report = {
            'model_name': model_name,
            'basic_metrics': self.calculate_basic_metrics(y_true, y_pred, y_prob),
            'business_metrics': self.calculate_business_metrics(y_true, y_pred, transaction_amounts),
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Store results
        self.evaluation_results[model_name] = report
        
        return report
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple models based on evaluation metrics.
        
        Args:
            model_results (Dict[str, Dict[str, Any]]): Results from multiple models
            
        Returns:
            pd.DataFrame: Comparison dataframe
        """
        comparison_data = []
        
        for model_name, results in model_results.items():
            if 'basic_metrics' in results:
                row = {'Model': model_name}
                row.update(results['basic_metrics'])
                if 'business_metrics' in results:
                    row.update(results['business_metrics'])
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by ROC AUC if available, otherwise by F1 score
        if 'roc_auc' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('roc_auc', ascending=False)
        elif 'f1_score' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('f1_score', ascending=False)
        
        return comparison_df


def main():
    """
    Example usage of the ModelEvaluator class.
    """
    try:
        # Create sample data for demonstration
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Generate sample data
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_classes=2,
            weights=[0.9, 0.1],
            random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train a sample model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Initialize evaluator
        evaluator = ModelEvaluator()
        
        # Generate evaluation report
        report = evaluator.generate_evaluation_report(
            y_test, y_pred, y_prob, model_name="Random Forest"
        )
        
        print("Evaluation Report:")
        print(f"Model: {report['model_name']}")
        print("\nBasic Metrics:")
        for metric, value in report['basic_metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nBusiness Metrics:")
        for metric, value in report['business_metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
        # Create visualizations
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Plot ROC curve
        roc_fig = evaluator.plot_roc_curve(y_test, y_prob, "Random Forest")
        plt.show()
        
        # Plot feature importance
        importance_fig = evaluator.plot_feature_importance(model, feature_names, top_n=10)
        plt.show()
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")


if __name__ == "__main__":
    main()