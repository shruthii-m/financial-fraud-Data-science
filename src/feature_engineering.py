"""
Feature Engineering Module for Financial Fraud Detection

This module contains functions for creating new features from the raw
financial transaction data to improve model performance.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    A class for engineering features from financial transaction data.
    """
    
    def __init__(self):
        self.customer_stats = {}
        self.merchant_stats = {}
        
    def create_time_features(self, df: pd.DataFrame, timestamp_col: str = 'step') -> pd.DataFrame:
        """
        Create time-based features from timestamp column.
        
        Args:
            df (pd.DataFrame): Input dataframe
            timestamp_col (str): Name of the timestamp column
            
        Returns:
            pd.DataFrame: Dataframe with time features
        """
        logger.info("Creating time-based features")
        
        df_features = df.copy()
        
        if timestamp_col in df.columns:
            # Convert step to hours (assuming each step is 1 hour)
            df_features['hour'] = df[timestamp_col] % 24
            df_features['day'] = df[timestamp_col] // 24
            df_features['week'] = df[timestamp_col] // (24 * 7)
            
            # Create time-based categorical features
            df_features['is_weekend'] = (df_features['day'] % 7).isin([5, 6]).astype(int)
            df_features['is_night'] = ((df_features['hour'] >= 22) | (df_features['hour'] <= 6)).astype(int)
            df_features['is_business_hours'] = ((df_features['hour'] >= 9) & (df_features['hour'] <= 17)).astype(int)
            
            logger.info("Time-based features created successfully")
        else:
            logger.warning(f"Timestamp column '{timestamp_col}' not found")
            
        return df_features
    
    def create_amount_features(self, df: pd.DataFrame, amount_col: str = 'amount') -> pd.DataFrame:
        """
        Create amount-based features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            amount_col (str): Name of the amount column
            
        Returns:
            pd.DataFrame: Dataframe with amount features
        """
        logger.info("Creating amount-based features")
        
        df_features = df.copy()
        
        if amount_col in df.columns:
            # Log transformation to handle skewness
            df_features['log_amount'] = np.log1p(df[amount_col])
            
            # Amount categories
            df_features['amount_category'] = pd.cut(
                df[amount_col], 
                bins=[0, 100, 1000, 10000, 100000, float('inf')], 
                labels=['very_low', 'low', 'medium', 'high', 'very_high']
            ).astype(str)
            
            # Round number indicator (suspicious for fraud)
            df_features['is_round_amount'] = (df[amount_col] % 100 == 0).astype(int)
            df_features['is_very_round_amount'] = (df[amount_col] % 1000 == 0).astype(int)
            
            logger.info("Amount-based features created successfully")
        else:
            logger.warning(f"Amount column '{amount_col}' not found")
            
        return df_features
    
    def create_balance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create balance-related features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with balance features
        """
        logger.info("Creating balance-related features")
        
        df_features = df.copy()
        
        # Balance difference features
        if 'oldbalanceOrg' in df.columns and 'newbalanceOrig' in df.columns:
            df_features['balance_change_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
            df_features['balance_change_orig_ratio'] = np.where(
                df['oldbalanceOrg'] != 0,
                df_features['balance_change_orig'] / df['oldbalanceOrg'],
                0
            )
        
        if 'oldbalanceDest' in df.columns and 'newbalanceDest' in df.columns:
            df_features['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
            df_features['balance_change_dest_ratio'] = np.where(
                df['oldbalanceDest'] != 0,
                df_features['balance_change_dest'] / df['oldbalanceDest'],
                0
            )
        
        # Balance vs amount ratios (fixed to avoid infinite values)
        if 'amount' in df.columns:
            if 'oldbalanceOrg' in df.columns:
                # Use a large but finite value instead of infinity
                df_features['amount_to_balance_orig_ratio'] = np.where(
                    df['oldbalanceOrg'] > 1e-10,  # Use small threshold to avoid division by tiny numbers
                    np.clip(df['amount'] / df['oldbalanceOrg'], -1e6, 1e6),  # Clip to prevent extreme values
                    0  # Set to 0 when balance is effectively zero
                )
            
            if 'oldbalanceDest' in df.columns:
                df_features['amount_to_balance_dest_ratio'] = np.where(
                    df['oldbalanceDest'] > 1e-10,  # Use small threshold
                    np.clip(df['amount'] / df['oldbalanceDest'], -1e6, 1e6),  # Clip to prevent extreme values
                    0  # Set to 0 when balance is effectively zero
                )
        
        # Zero balance indicators
        if 'oldbalanceOrg' in df.columns:
            df_features['orig_zero_balance_before'] = (df['oldbalanceOrg'] == 0).astype(int)
        if 'newbalanceOrig' in df.columns:
            df_features['orig_zero_balance_after'] = (df['newbalanceOrig'] == 0).astype(int)
        if 'oldbalanceDest' in df.columns:
            df_features['dest_zero_balance_before'] = (df['oldbalanceDest'] == 0).astype(int)
        if 'newbalanceDest' in df.columns:
            df_features['dest_zero_balance_after'] = (df['newbalanceDest'] == 0).astype(int)
        
        logger.info("Balance-related features created successfully")
        return df_features
    
    def create_transaction_type_features(self, df: pd.DataFrame, type_col: str = 'type') -> pd.DataFrame:
        """
        Create features based on transaction type.
        
        Args:
            df (pd.DataFrame): Input dataframe
            type_col (str): Name of the transaction type column
            
        Returns:
            pd.DataFrame: Dataframe with transaction type features
        """
        logger.info("Creating transaction type features")
        
        df_features = df.copy()
        
        if type_col in df.columns:
            # One-hot encoding for transaction types
            type_dummies = pd.get_dummies(df[type_col], prefix='type')
            df_features = pd.concat([df_features, type_dummies], axis=1)
            
            # High-risk transaction type indicator
            high_risk_types = ['TRANSFER', 'CASH_OUT']
            df_features['is_high_risk_type'] = df[type_col].isin(high_risk_types).astype(int)
            
            logger.info("Transaction type features created successfully")
        else:
            logger.warning(f"Transaction type column '{type_col}' not found")
            
        return df_features
    
    def create_customer_features(self, df: pd.DataFrame, customer_col: str = 'nameOrig') -> pd.DataFrame:
        """
        Create customer-based aggregated features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            customer_col (str): Name of the customer column
            
        Returns:
            pd.DataFrame: Dataframe with customer features
        """
        logger.info("Creating customer-based features")
        
        df_features = df.copy()
        
        if customer_col in df.columns and 'amount' in df.columns:
            # Calculate customer statistics
            customer_stats = df.groupby(customer_col).agg({
                'amount': ['count', 'sum', 'mean', 'std', 'min', 'max'],
                'step': ['min', 'max']
            }).reset_index()
            
            # Flatten column names
            customer_stats.columns = [customer_col] + [
                f'customer_{stat[1]}_{stat[0]}' if stat[1] else f'customer_{stat[0]}'
                for stat in customer_stats.columns[1:]
            ]
            
            # Calculate additional features
            customer_stats['customer_transaction_span'] = (
                customer_stats['customer_max_step'] - customer_stats['customer_min_step']
            )
            customer_stats['customer_avg_transaction_frequency'] = np.where(
                customer_stats['customer_transaction_span'] > 0,
                customer_stats['customer_count_amount'] / customer_stats['customer_transaction_span'],
                0
            )
            
            # Merge back to original dataframe
            df_features = df_features.merge(customer_stats, on=customer_col, how='left')
            
            # Create relative features
            df_features['amount_vs_customer_avg'] = np.where(
                df_features['customer_mean_amount'] != 0,
                df_features['amount'] / df_features['customer_mean_amount'],
                1
            )
            
            df_features['amount_vs_customer_max'] = np.where(
                df_features['customer_max_amount'] != 0,
                df_features['amount'] / df_features['customer_max_amount'],
                1
            )
            
            logger.info("Customer-based features created successfully")
        else:
            logger.warning(f"Required columns not found for customer features")
            
        return df_features
    
    def create_merchant_features(self, df: pd.DataFrame, merchant_col: str = 'nameDest') -> pd.DataFrame:
        """
        Create merchant-based aggregated features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            merchant_col (str): Name of the merchant column
            
        Returns:
            pd.DataFrame: Dataframe with merchant features
        """
        logger.info("Creating merchant-based features")
        
        df_features = df.copy()
        
        if merchant_col in df.columns and 'amount' in df.columns:
            # Calculate merchant statistics
            merchant_stats = df.groupby(merchant_col).agg({
                'amount': ['count', 'sum', 'mean', 'std'],
                'step': ['min', 'max']
            }).reset_index()
            
            # Flatten column names
            merchant_stats.columns = [merchant_col] + [
                f'merchant_{stat[1]}_{stat[0]}' if stat[1] else f'merchant_{stat[0]}'
                for stat in merchant_stats.columns[1:]
            ]
            
            # Merge back to original dataframe
            df_features = df_features.merge(merchant_stats, on=merchant_col, how='left')
            
            # Create relative features
            df_features['amount_vs_merchant_avg'] = np.where(
                df_features['merchant_mean_amount'] != 0,
                df_features['amount'] / df_features['merchant_mean_amount'],
                1
            )
            
            logger.info("Merchant-based features created successfully")
        else:
            logger.warning(f"Required columns not found for merchant features")
            
        return df_features
    
    def _validate_and_clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean engineered features to prevent infinite/large values.
        
        Args:
            df (pd.DataFrame): DataFrame with engineered features
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        logger.info("Validating and cleaning engineered features")
        
        df_clean = df.copy()
        
        # Get numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Count problematic values before cleaning
            inf_count = np.isinf(df_clean[col]).sum()
            nan_count = np.isnan(df_clean[col]).sum()
            
            if inf_count > 0 or nan_count > 0:
                logger.warning(f"Found {inf_count} infinite and {nan_count} NaN values in {col}")
                
                # Replace infinite values with NaN first
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
                
                # Fill NaN with median, or 0 if all values are NaN
                if df_clean[col].notna().any():
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
                else:
                    df_clean[col] = df_clean[col].fillna(0)
            
            # Check for extremely large values and clip them
            if df_clean[col].abs().max() > 1e6:
                logger.warning(f"Clipping extremely large values in {col}")
                df_clean[col] = df_clean[col].clip(-1e6, 1e6)
        
        logger.info("Feature validation and cleaning completed")
        return df_clean
    
    def create_fraud_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create fraud indicator features based on domain knowledge.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with fraud indicator features
        """
        logger.info("Creating fraud indicator features")
        
        df_features = df.copy()
        
        # Error in balance calculations (suspicious)
        if all(col in df.columns for col in ['oldbalanceOrg', 'newbalanceOrig', 'amount']):
            expected_balance = df['oldbalanceOrg'] - df['amount']
            df_features['balance_error_orig'] = (
                df['newbalanceOrig'] != expected_balance
            ).astype(int)
        
        if all(col in df.columns for col in ['oldbalanceDest', 'newbalanceDest', 'amount']):
            expected_balance = df['oldbalanceDest'] + df['amount']
            df_features['balance_error_dest'] = (
                df['newbalanceDest'] != expected_balance
            ).astype(int)
        
        # Same origin and destination (suspicious)
        if 'nameOrig' in df.columns and 'nameDest' in df.columns:
            df_features['same_orig_dest'] = (
                df['nameOrig'] == df['nameDest']
            ).astype(int)
        
        # Large amount transactions
        if 'amount' in df.columns:
            amount_threshold = df['amount'].quantile(0.95)
            df_features['is_large_amount'] = (
                df['amount'] > amount_threshold
            ).astype(int)
        
        logger.info("Fraud indicator features created successfully")
        return df_features
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        logger.info("Starting complete feature engineering pipeline")
        
        # Apply all feature engineering methods
        df_features = df.copy()
        
        df_features = self.create_time_features(df_features)
        df_features = self.create_amount_features(df_features)
        df_features = self.create_balance_features(df_features)
        df_features = self.create_transaction_type_features(df_features)
        df_features = self.create_fraud_indicators(df_features)
        
        # Note: Customer and merchant features require careful handling
        # to avoid data leakage in time-series data
        # df_features = self.create_customer_features(df_features)
        # df_features = self.create_merchant_features(df_features)
        
        # Validate and clean all engineered features to prevent infinite/large values
        df_features = self._validate_and_clean_features(df_features)
        
        logger.info(f"Feature engineering completed. New shape: {df_features.shape}")
        return df_features


def main():
    """
    Example usage of the FeatureEngineer class.
    """
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    try:
        # Example: Load preprocessed data
        # df = pd.read_csv("../data/processed/preprocessed_data.csv")
        
        # Create sample data for demonstration
        sample_data = pd.DataFrame({
            'step': [1, 2, 3, 24, 25],
            'type': ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'PAYMENT', 'TRANSFER'],
            'amount': [100.0, 500.0, 1000.0, 50.0, 200.0],
            'nameOrig': ['C1', 'C2', 'C1', 'C3', 'C1'],
            'oldbalanceOrg': [1000, 2000, 900, 500, 700],
            'newbalanceOrig': [900, 1500, 0, 450, 500],
            'nameDest': ['M1', 'M2', 'M3', 'M1', 'M2'],
            'oldbalanceDest': [0, 0, 0, 100, 200],
            'newbalanceDest': [100, 500, 1000, 150, 400]
        })
        
        # Engineer features
        df_engineered = engineer.engineer_features(sample_data)
        
        print(f"Original features: {sample_data.shape[1]}")
        print(f"Engineered features: {df_engineered.shape[1]}")
        print(f"New features added: {df_engineered.shape[1] - sample_data.shape[1]}")
        
        # Display new feature columns
        new_features = [col for col in df_engineered.columns if col not in sample_data.columns]
        print(f"New features: {new_features}")
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")


if __name__ == "__main__":
    main()