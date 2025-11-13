"""
Data Preprocessing Module for Financial Fraud Detection

This module contains functions for loading, cleaning, and preprocessing
the financial transaction data from the Kaggle dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    A class for preprocessing financial transaction data for fraud detection.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        logger.info("Handling missing values")
        
        # Check for missing values
        missing_summary = df.isnull().sum()
        if missing_summary.sum() > 0:
            logger.info(f"Missing values found:\n{missing_summary[missing_summary > 0]}")
            
            # Handle numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True)
            
            # Handle categorical columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].mode()[0], inplace=True)
        
        logger.info("Missing values handled successfully")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using label encoding.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical features
        """
        logger.info("Encoding categorical features")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col not in ['isFraud', 'nameOrig', 'nameDest']:  # Skip target and high cardinality columns
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"Encoded column: {col}")
        
        return df_encoded
    
    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers from numerical columns.
        
        Args:
            df (pd.DataFrame): Input dataframe
            method (str): Method for outlier detection ('iqr' or 'zscore')
            threshold (float): Threshold for outlier detection
            
        Returns:
            pd.DataFrame: Dataframe with outliers removed
        """
        logger.info(f"Removing outliers using {method} method")
        
        df_clean = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > threshold
            
            outlier_count = outliers.sum()
            if outlier_count > 0:
                logger.info(f"Removed {outlier_count} outliers from {col}")
                df_clean = df_clean[~outliers]
        
        return df_clean
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> tuple:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features (optional)
            
        Returns:
            tuple: Scaled training and test features
        """
        logger.info("Scaling features")
        
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns
        
        X_train_scaled = X_train.copy()
        X_train_scaled[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        
        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled, None
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'isFraud', 
                    test_size: float = 0.2, random_state: int = 42) -> tuple:
        """
        Complete data preprocessing pipeline.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of the target column
            test_size (float): Size of the test set
            random_state (int): Random state for reproducibility
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        logger.info("Starting complete data preprocessing pipeline")
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Remove high cardinality columns that might cause overfitting
        columns_to_drop = ['nameOrig', 'nameDest']
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        if existing_columns:
            df = df.drop(columns=existing_columns)
            logger.info(f"Dropped high cardinality columns: {existing_columns}")
        
        # Separate features and target
        if target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
        else:
            logger.warning(f"Target column '{target_column}' not found. Using all columns as features.")
            X = df
            y = None
        
        self.feature_columns = X.columns.tolist()
        
        # Split the data
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Scale features
            X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
            
            logger.info(f"Data preprocessing completed. Training set size: {X_train_scaled.shape}")
            return X_train_scaled, X_test_scaled, y_train, y_test
        else:
            # If no target column, just scale the features
            X_scaled, _ = self.scale_features(X)
            return X_scaled, None, None, None


def main():
    """
    Example usage of the DataPreprocessor class.
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load and preprocess data
    try:
        # Example file path - update with actual path
        data_path = "../data/raw/fraud_detection_dataset.csv"
        df = preprocessor.load_data(data_path)
        
        # Prepare data for modeling
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Class distribution in training set:")
        print(y_train.value_counts(normalize=True))
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")


if __name__ == "__main__":
    main()