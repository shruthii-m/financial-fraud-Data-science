"""
Machine Learning Modeling Module for Financial Fraud Detection

This module contains various machine learning models and ensemble methods
for detecting fraudulent financial transactions.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import logging
from typing import Dict, Any, Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDetectionModels:
    """
    A class containing various machine learning models for fraud detection.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_params = {}
        
    def get_base_models(self) -> Dict[str, Any]:
        """
        Get dictionary of base machine learning models.
        
        Returns:
            Dict[str, Any]: Dictionary of model names and instances
        """
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            'svm': SVC(
                random_state=self.random_state,
                class_weight='balanced',
                probability=True
            ),
            'naive_bayes': GaussianNB(),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=self.random_state,
                max_iter=500
            )
        }
        return models
    
    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series, 
                             method: str = 'smote') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance using various techniques.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            method (str): Method for handling imbalance ('smote', 'undersample', 'none')
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Resampled features and target
        """
        logger.info(f"Handling class imbalance using {method}")
        
        original_distribution = y.value_counts(normalize=True)
        logger.info(f"Original class distribution:\n{original_distribution}")
        
        # Clean data before resampling
        X_clean = self._clean_training_data(X)
        
        if method == 'smote':
            smote = SMOTE(random_state=self.random_state)
            X_resampled, y_resampled = smote.fit_resample(X_clean, y)
        elif method == 'undersample':
            undersampler = RandomUnderSampler(random_state=self.random_state)
            X_resampled, y_resampled = undersampler.fit_resample(X_clean, y)
        else:
            X_resampled, y_resampled = X_clean, y
        
        if method != 'none':
            new_distribution = pd.Series(y_resampled).value_counts(normalize=True)
            logger.info(f"New class distribution:\n{new_distribution}")
        
        return X_resampled, y_resampled
    
    def train_single_model(self, model_name: str, model: Any, X_train: pd.DataFrame, 
                          y_train: pd.Series, cv_folds: int = 5) -> Dict[str, float]:
        """
        Train a single model and evaluate using cross-validation.
        
        Args:
            model_name (str): Name of the model
            model (Any): Model instance
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            Dict[str, float]: Dictionary with model performance metrics
        """
        logger.info(f"Training {model_name}")
        
        # Clean data before training to prevent infinity/large value errors
        X_train_clean = self._clean_training_data(X_train)
        
        # Stratified K-Fold cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Cross-validation scores using cleaned data
        cv_scores = cross_val_score(model, X_train_clean, y_train, cv=skf, scoring='roc_auc')
        
        # Train the model on full training set using cleaned data
        model.fit(X_train_clean, y_train)
        
        # Store the trained model
        self.models[model_name] = model
        
        results = {
            'cv_mean_auc': cv_scores.mean(),
            'cv_std_auc': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
        
        logger.info(f"{model_name} - CV AUC: {results['cv_mean_auc']:.4f} (+/- {results['cv_std_auc']:.4f})")
        
        return results
    
    def _clean_training_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Clean training data to remove infinite values and extremely large numbers.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Cleaned features
        """
        logger.info("Cleaning training data for infinite/large values")
        
        X_clean = X.copy()
        
        # Replace infinite values with NaN
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        # Handle very large values by capping at reasonable limits
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Cap extreme values at 99.9th and 0.1th percentiles
            if X_clean[col].notna().any():  # Check if column has non-NaN values
                upper_cap = X_clean[col].quantile(0.999)
                lower_cap = X_clean[col].quantile(0.001)
                
                # Only cap if we have valid quantiles
                if not (np.isnan(upper_cap) or np.isnan(lower_cap)):
                    X_clean[col] = X_clean[col].clip(lower=lower_cap, upper=upper_cap)
        
        # Fill NaN values with median or 0 if median is not available
        for col in numeric_cols:
            if X_clean[col].isnull().any():
                median_val = X_clean[col].median()
                fill_val = median_val if not np.isnan(median_val) else 0
                X_clean[col] = X_clean[col].fillna(fill_val)
        
        # Final safety check - replace any remaining non-finite values
        X_clean = X_clean.replace([np.inf, -np.inf], 0)
        X_clean = X_clean.fillna(0)
        
        # Check for extremely large values and cap them to safe float64 range
        for col in numeric_cols:
            # Cap values to a safe range for float64 calculations
            max_safe_value = 1e15  # Conservative limit for numerical stability
            X_clean[col] = X_clean[col].clip(-max_safe_value, max_safe_value)
        
        # Additional check: ensure no remaining problematic values
        if np.isinf(X_clean).any().any() or np.isnan(X_clean).any().any():
            logger.warning("Some infinite or NaN values remain after cleaning, applying final cleanup")
            X_clean = X_clean.replace([np.inf, -np.inf, np.nan], 0)
        
        logger.info(f"Data cleaning completed. Shape: {X_clean.shape}")
        
        # Final validation - log any issues found and fixed
        try:
            numeric_cols = X.select_dtypes(include=[np.number]).columns if hasattr(X, 'select_dtypes') else X.columns
            inf_count = np.isinf(X[numeric_cols]).sum().sum() if len(numeric_cols) > 0 else 0
            nan_count = np.isnan(X[numeric_cols]).sum().sum() if len(numeric_cols) > 0 else 0
            if inf_count > 0 or nan_count > 0:
                logger.info(f"Cleaned {inf_count} infinite values and {nan_count} NaN values")
        except Exception:
            # Skip validation if there are type issues
            pass
        
        return X_clean
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        resample_method: str = 'smote', cv_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Train all base models and compare their performance.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            resample_method (str): Method for handling class imbalance
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            Dict[str, Dict[str, float]]: Results for all models
        """
        logger.info("Training all base models")
        
        # Handle class imbalance
        X_resampled, y_resampled = self.handle_class_imbalance(X_train, y_train, resample_method)
        
        # Get base models
        base_models = self.get_base_models()
        
        # Train each model
        all_results = {}
        for model_name, model in base_models.items():
            try:
                results = self.train_single_model(model_name, model, X_resampled, y_resampled, cv_folds)
                all_results[model_name] = results
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                all_results[model_name] = {'error': str(e)}
        
        # Find best model
        best_model_name = max(
            [name for name, results in all_results.items() if 'cv_mean_auc' in results],
            key=lambda x: all_results[x]['cv_mean_auc']
        )
        self.best_model = self.models[best_model_name]
        logger.info(f"Best model: {best_model_name}")
        
        return all_results
    
    def create_ensemble_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                            model_names: List[str] = None) -> VotingClassifier:
        """
        Create an ensemble model using voting classifier.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            model_names (List[str]): List of model names to include in ensemble
            
        Returns:
            VotingClassifier: Trained ensemble model
        """
        logger.info("Creating ensemble model")
        
        if model_names is None:
            model_names = ['random_forest', 'gradient_boosting', 'logistic_regression']
        
        # Select models for ensemble
        ensemble_models = []
        for name in model_names:
            if name in self.models:
                ensemble_models.append((name, self.models[name]))
        
        if len(ensemble_models) < 2:
            logger.warning("Not enough models for ensemble. Training base models first.")
            self.train_all_models(X_train, y_train)
            ensemble_models = [(name, self.models[name]) for name in model_names if name in self.models]
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'
        )
        
        # Clean data and train ensemble
        X_train_clean = self._clean_training_data(X_train)
        ensemble.fit(X_train_clean, y_train)
        self.models['ensemble'] = ensemble
        
        logger.info("Ensemble model created and trained")
        return ensemble
    
    def hyperparameter_tuning(self, model_name: str, X_train: pd.DataFrame, 
                            y_train: pd.Series, param_grids: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            model_name (str): Name of the model to tune
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            param_grids (Dict[str, Any]): Parameter grids for tuning
            
        Returns:
            Dict[str, Any]: Best parameters and score
        """
        logger.info(f"Hyperparameter tuning for {model_name}")
        
        if param_grids is None:
            param_grids = self._get_default_param_grids()
        
        if model_name not in param_grids:
            logger.warning(f"No parameter grid defined for {model_name}")
            return {}
        
        # Get base model
        base_models = self.get_base_models()
        if model_name not in base_models:
            logger.error(f"Model {model_name} not found")
            return {}
        
        model = base_models[model_name]
        param_grid = param_grids[model_name]
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state),
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        # Clean and handle class imbalance
        X_resampled, y_resampled = self.handle_class_imbalance(X_train, y_train, 'smote')
        
        # Fit grid search
        grid_search.fit(X_resampled, y_resampled)
        
        # Store best model and parameters
        self.models[f"{model_name}_tuned"] = grid_search.best_estimator_
        self.best_params[model_name] = grid_search.best_params_
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_
        }
        
        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return results
    
    def _get_default_param_grids(self) -> Dict[str, Dict[str, List]]:
        """
        Get default parameter grids for hyperparameter tuning.
        
        Returns:
            Dict[str, Dict[str, List]]: Parameter grids for each model
        """
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7]
            },
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        }
        return param_grids
    
    def save_model(self, model_name: str, file_path: str) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model_name (str): Name of the model to save
            file_path (str): Path to save the model
        """
        if model_name in self.models:
            joblib.dump(self.models[model_name], file_path)
            logger.info(f"Model {model_name} saved to {file_path}")
        else:
            logger.error(f"Model {model_name} not found")
    
    def load_model(self, file_path: str, model_name: str = None) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            file_path (str): Path to the saved model
            model_name (str): Name to assign to the loaded model
            
        Returns:
            Any: Loaded model
        """
        try:
            model = joblib.load(file_path)
            if model_name:
                self.models[model_name] = model
            logger.info(f"Model loaded from {file_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None


def main():
    """
    Example usage of the FraudDetectionModels class.
    """
    try:
        # Initialize model trainer
        model_trainer = FraudDetectionModels(random_state=42)
        
        # Create sample data for demonstration
        from sklearn.datasets import make_classification
        
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_classes=2,
            weights=[0.95, 0.05],  # Imbalanced dataset
            random_state=42
        )
        
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y_series = pd.Series(y)
        
        print("Training all models...")
        results = model_trainer.train_all_models(X_df, y_series)
        
        print("\nModel Performance Summary:")
        for model_name, metrics in results.items():
            if 'cv_mean_auc' in metrics:
                print(f"{model_name}: {metrics['cv_mean_auc']:.4f} (+/- {metrics['cv_std_auc']:.4f})")
        
        # Create ensemble model
        print("\nCreating ensemble model...")
        ensemble = model_trainer.create_ensemble_model(X_df, y_series)
        
        # Example of hyperparameter tuning
        print("\nPerforming hyperparameter tuning for Random Forest...")
        tuning_results = model_trainer.hyperparameter_tuning('random_forest', X_df, y_series)
        
        if tuning_results:
            print(f"Best parameters: {tuning_results['best_params']}")
            print(f"Best score: {tuning_results['best_score']:.4f}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")


if __name__ == "__main__":
    main()