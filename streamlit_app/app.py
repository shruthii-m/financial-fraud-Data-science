"""
Streamlit Web Application for Financial Fraud Detection

This application provides an interactive interface for fraud detection,
model evaluation, and data exploration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import sys
import os
from datetime import datetime
import warnings

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import custom modules
try:
    from preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer
    from modeling import FraudDetectionModels
    from evaluation import ModelEvaluator
except ImportError:
    st.error("Unable to import custom modules. Please ensure the src directory is properly configured.")

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Financial Fraud Detection System",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

class FraudDetectionApp:
    """Main application class for the Streamlit fraud detection app."""
    
    def __init__(self):
        # Check if data exists in session state and load it
        self.data = st.session_state.get('data', None)
        self.preprocessor = None
        self.feature_engineer = None
        self.model_trainer = None
        self.evaluator = None
        self.models = {}
        self.data_info = {}  # Store metadata about loaded data
    
    def get_memory_usage(self, df):
        """Calculate memory usage of dataframe in MB."""
        return df.memory_usage(deep=True).sum() / 1024 / 1024
    
    def suggest_sample_size(self, file_path):
        """Suggest appropriate sample size based on file size."""
        try:
            file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
            if file_size < 50:
                return "Full dataset"
            elif file_size < 200:
                return "500K rows"
            elif file_size < 500:
                return "100K rows"
            else:
                return "50K rows"
        except:
            return "100K rows"
        
    def load_sample_data(self, n_samples=1000):
        """Load sample fraud detection data for demonstration."""
        # Create sample data similar to the Kaggle dataset structure
        np.random.seed(42)
        
        data = {
            'step': np.random.randint(1, 744, n_samples),  # 1 month in hours
            'type': np.random.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'], n_samples, 
                                   p=[0.4, 0.2, 0.2, 0.1, 0.1]),
            'amount': np.random.lognormal(5, 2, n_samples),
            'nameOrig': [f'C{i}' for i in np.random.randint(1, 100, n_samples)],
            'oldbalanceOrg': np.random.lognormal(8, 2, n_samples),
            'newbalanceOrig': np.random.lognormal(8, 2, n_samples),
            'nameDest': [f'M{i}' for i in np.random.randint(1, 50, n_samples)],
            'oldbalanceDest': np.random.lognormal(8, 2, n_samples),
            'newbalanceDest': np.random.lognormal(8, 2, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create fraud labels (5% fraud rate)
        fraud_mask = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
        df['isFraud'] = fraud_mask
        
        # Make fraud cases more realistic
        fraud_indices = df[df['isFraud'] == 1].index
        df.loc[fraud_indices, 'type'] = np.random.choice(['TRANSFER', 'CASH_OUT'], len(fraud_indices))
        df.loc[fraud_indices, 'amount'] = np.random.lognormal(7, 1.5, len(fraud_indices))
        
        return df
    
    def sidebar_navigation(self):
        """Create sidebar navigation."""
        st.sidebar.title("�� Navigation")
        
        pages = {
            "�� Home": "home",
            "�� Data Explorer": "data_explorer",
            "�� Data Preprocessing": "preprocessing", 
            "⚙️ Feature Engineering": "feature_engineering",
            "�� Model Training": "model_training",
            "�� Model Evaluation": "evaluation",
            "�� Prediction": "prediction"
        }
        
        selected_page = st.sidebar.selectbox("Select Page", list(pages.keys()))
        return pages[selected_page]
    
    def home_page(self):
        """Display the home page."""
        st.title("�� Financial Fraud Detection System")
        
        st.markdown("""
        ## Welcome to the Financial Fraud Detection System
        
        This comprehensive system helps detect fraudulent financial transactions using machine learning techniques.
        
        ### Features:
        - **Data Exploration**: Analyze transaction patterns and distributions
        - **Data Preprocessing**: Clean and prepare data for modeling
        - **Feature Engineering**: Create meaningful features from raw transaction data
        - **Model Training**: Train various machine learning models
        - **Model Evaluation**: Comprehensive evaluation with business metrics
        - **Real-time Prediction**: Make predictions on new transactions
        
        ### Getting Started:
        1. Start by exploring your data in the **Data Explorer**
        2. Preprocess your data to handle missing values and outliers
        3. Engineer features to improve model performance
        4. Train multiple models and compare their performance
        5. Evaluate models using comprehensive metrics
        6. Use the best model for real-time predictions
        
        ### Dataset Information:
        This system is designed to work with the **Financial Fraud Detection dataset by Aman Ali Siddiqui** from Kaggle.
        The dataset contains financial transaction records with features like transaction type, amount, 
        origin/destination balances, and fraud labels.
        
        ### �� Large Dataset (470MB) Loading Guide:
        **Option 1: File Path Method (Recommended)**
        1. Download the Kaggle dataset to your computer
        2. Go to **Data Explorer** → **Local File Path** tab
        3. Enter the full path to your CSV file
        4. Choose sample size or use chunked reading
        
        **Option 2: Raw Folder Method**
        1. Copy CSV to: `data/raw/fraud_detection_dataset.csv`
        2. Go to **Data Explorer** → **Load from Raw Folder** tab
        3. Click "Load from Raw Folder"
        
        **Option 3: Start with Sample Data**
        1. Use **Generate Sample Data** for testing
        2. Experiment with different features
        3. Switch to real data when ready
        """)
        
        # Display dataset statistics if data is loaded
        if self.data is not None:
            st.subheader("�� Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Transactions", len(self.data))
            with col2:
                fraud_count = self.data['isFraud'].sum() if 'isFraud' in self.data.columns else 0
                st.metric("Fraud Cases", fraud_count)
            with col3:
                fraud_rate = (fraud_count / len(self.data)) * 100 if len(self.data) > 0 else 0
                st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
            with col4:
                total_amount = self.data['amount'].sum() if 'amount' in self.data.columns else 0
                st.metric("Total Amount", f"${total_amount:,.2f}")
    
    def data_explorer_page(self):
        """Display data exploration page."""
        st.title("�� Data Explorer")
        
        # Data loading section
        st.subheader("�� Data Loading Options")
        
        # Add tabs for different loading methods
        tab1, tab2, tab3, tab4 = st.tabs(["�� Local File Path", "☁️ Upload Small File", "�� Sample Data", "⚡ Load from Raw Folder"])
        
        with tab1:
            st.write("**Load Large Dataset from File Path** (Recommended for 470MB+ files)")
            file_path = st.text_input(
                "Enter full path to your CSV file:", 
                placeholder="e.g., C:/Users/YourName/Downloads/fraud_detection.csv",
                help="This is the best option for large files like the Kaggle dataset (470MB)"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                sample_size = st.selectbox(
                    "Sample size for initial exploration:", 
                    ["Full dataset", "1M rows", "500K rows", "100K rows", "50K rows"],
                    index=3,
                    help="Start with smaller sample for faster exploration"
                )
            with col2:
                use_chunks = st.checkbox("Use chunked reading", value=True, help="Loads data in chunks to avoid memory issues")
            
            if st.button("Load from Path", type="primary"):
                if file_path:
                    try:
                        with st.spinner(f"Loading data from {file_path}..."):
                            # Determine number of rows to load
                            nrows = None
                            if sample_size != "Full dataset":
                                nrows = int(sample_size.split()[0].replace('K', '000').replace('M', '000000'))
                            
                            if use_chunks and nrows is None:
                                # Read in chunks for very large files
                                chunk_size = 50000
                                chunks = []
                                total_chunks = 0
                                progress_bar = st.progress(0)
                                
                                for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                                    chunks.append(chunk)
                                    total_chunks += 1
                                    progress_bar.progress(min(total_chunks * chunk_size / 100000, 1.0))
                                    if total_chunks >= 20:  # Limit to ~1M rows for memory
                                        break
                                
                                self.data = pd.concat(chunks, ignore_index=True)
                                progress_bar.empty()
                            else:
                                # Read normally
                                self.data = pd.read_csv(file_path, nrows=nrows)
                            
                            # Store in session state for persistence across pages
                            st.session_state['data'] = self.data
                            st.session_state['data_loaded'] = True
                            
                            st.success(f"✅ Data loaded successfully! Shape: {self.data.shape}")
                            st.info(f"�� Memory usage: {self.data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
                            
                    except FileNotFoundError:
                        st.error("❌ File not found. Please check the file path.")
                    except Exception as e:
                        st.error(f"❌ Error loading file: {str(e)}")
                        st.write("**Troubleshooting tips:**")
                        st.write("- Make sure the file path is correct")
                        st.write("- Try with a smaller sample size first")
                        st.write("- Ensure the file is not open in another program")
                else:
                    st.warning("Please enter a file path")
        
        with tab2:
            st.write("**Upload Small Files** (< 200MB)")
            st.warning("⚠️ Large files (470MB+) may cause browser issues. Use 'Local File Path' instead.")
            
            uploaded_file = st.file_uploader(
                "Choose CSV file", 
                type=['csv'], 
                help="For files larger than 200MB, use the 'Local File Path' option"
            )
            if uploaded_file is not None:
                try:
                    # Check file size
                    file_size = len(uploaded_file.getvalue()) / 1024 / 1024  # MB
                    st.write(f"File size: {file_size:.1f} MB")
                    
                    if file_size > 200:
                        st.error("⚠️ File too large for upload. Please use 'Local File Path' option instead.")
                    else:
                        with st.spinner("Processing uploaded file..."):
                            self.data = pd.read_csv(uploaded_file)
                            # Store in session state for persistence across pages
                            st.session_state['data'] = self.data
                            st.session_state['data_loaded'] = True
                            st.success(f"✅ Data uploaded successfully! Shape: {self.data.shape}")
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
        
        with tab3:
            st.write("**Generate Sample Data** (For testing and demonstration)")
            sample_size = st.selectbox("Sample size:", ["1K", "10K", "50K", "100K"], index=1)
            
            if st.button("Generate Sample Data"):
                with st.spinner("Generating sample data..."):
                    size = int(sample_size.replace('K', '000'))
                    self.data = self.load_sample_data(size)
                    # Store in session state for persistence across pages
                    st.session_state['data'] = self.data
                    st.session_state['data_loaded'] = True
                    st.success(f"✅ Sample data generated! Shape: {self.data.shape}")
        
        with tab4:
            st.write("**Quick Load from Raw Data Folder**")
            st.write("Place your CSV file in: `data/raw/fraud_detection_dataset.csv`")
            
            expected_path = "data/raw/fraud_detection_dataset.csv"
            if st.button("Load from Raw Folder"):
                try:
                    if os.path.exists(expected_path):
                        with st.spinner("Loading data from raw folder..."):
                            # Load first 100K rows for quick exploration
                            self.data = pd.read_csv(expected_path, nrows=100000)
                            # Store in session state for persistence across pages
                            st.session_state['data'] = self.data
                            st.session_state['data_loaded'] = True
                            st.success(f"✅ Data loaded from raw folder! Shape: {self.data.shape}")
                            st.info("�� Showing first 100K rows for performance. Use 'Local File Path' for full dataset.")
                    else:
                        st.error(f"❌ File not found at: {expected_path}")
                        st.write("�� **Setup Instructions:**")
                        st.write("1. Create folder: `data/raw/`")
                        st.write("2. Copy your CSV file there")
                        st.write("3. Rename it to: `fraud_detection_dataset.csv`")
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
        
        # Check if data is loaded in session state
        if not st.session_state.get('data_loaded', False) or self.data is None:
            st.info("Please load data to continue with exploration.")
            return
        
        # Dataset overview
        st.subheader("�� Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Info:**")
            st.write(f"- Shape: {self.data.shape}")
            st.write(f"- Missing values: {self.data.isnull().sum().sum()}")
            st.write(f"- Memory usage: {self.data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        with col2:
            st.write("**Data Types:**")
            dtype_counts = self.data.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                st.write(f"- {dtype}: {count} columns")
        
        # Display first few rows
        st.subheader("�� Data Sample")
        st.dataframe(self.data.head(10), width="stretch")
        
        # Statistical summary
        st.subheader("�� Statistical Summary")
        st.dataframe(self.data.describe(), width="stretch")
        
        # Visualizations
        if 'isFraud' in self.data.columns:
            st.subheader("�� Data Visualizations")
            
            # Fraud distribution
            fig_fraud = px.pie(
                values=self.data['isFraud'].value_counts().values,
                names=['Legitimate', 'Fraud'],
                title="Transaction Distribution"
            )
            st.plotly_chart(fig_fraud, width="stretch")
            
            # Transaction type distribution
            if 'type' in self.data.columns:
                fig_type = px.histogram(
                    self.data, x='type', color='isFraud',
                    title="Transaction Types by Fraud Status",
                    color_discrete_map={0: 'blue', 1: 'red'}
                )
                st.plotly_chart(fig_type, width="stretch")
            
            # Amount distribution
            if 'amount' in self.data.columns:
                fig_amount = px.box(
                    self.data, x='isFraud', y='amount',
                    title="Transaction Amount Distribution by Fraud Status"
                )
                fig_amount.update_yaxes(type="log")
                st.plotly_chart(fig_amount, width="stretch")
    
    def preprocessing_page(self):
        """Display data preprocessing page."""
        st.title("�� Data Preprocessing")
        
        # Check if data is loaded in session state
        if not st.session_state.get('data_loaded', False) or self.data is None:
            st.warning("Please load data first in the Data Explorer.")
            return
        
        st.subheader("⚙️ Preprocessing Options")
        
        # Initialize preprocessor
        if self.preprocessor is None:
            self.preprocessor = DataPreprocessor()
        
        col1, col2 = st.columns(2)
        
        with col1:
            handle_missing = st.checkbox("Handle Missing Values", value=True)
            encode_categorical = st.checkbox("Encode Categorical Features", value=True)
            remove_outliers = st.checkbox("Remove Outliers", value=False)
        
        with col2:
            if remove_outliers:
                outlier_method = st.selectbox("Outlier Detection Method", ['iqr', 'zscore'])
                outlier_threshold = st.slider("Outlier Threshold", 1.0, 3.0, 1.5, 0.1)
            scale_features = st.checkbox("Scale Features", value=True)
        
        if st.button("Run Preprocessing"):
            with st.spinner("Preprocessing data..."):
                try:
                    processed_data = self.data.copy()
                    
                    # Handle missing values
                    if handle_missing:
                        processed_data = self.preprocessor.handle_missing_values(processed_data)
                        st.success("✅ Missing values handled")
                    
                    # Encode categorical features
                    if encode_categorical:
                        processed_data = self.preprocessor.encode_categorical_features(processed_data)
                        st.success("✅ Categorical features encoded")
                    
                    # Remove outliers
                    if remove_outliers:
                        processed_data = self.preprocessor.remove_outliers(
                            processed_data, outlier_method, outlier_threshold
                        )
                        st.success("✅ Outliers removed")
                    
                    # Store processed data
                    st.session_state['processed_data'] = processed_data
                    
                    st.success("�� Preprocessing completed successfully!")
                    
                    # Show before/after comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Original Data:**")
                        st.write(f"Shape: {self.data.shape}")
                    with col2:
                        st.write("**Processed Data:**")
                        st.write(f"Shape: {processed_data.shape}")
                    
                except Exception as e:
                    st.error(f"Error during preprocessing: {str(e)}")
        
        # Display processed data if available
        if 'processed_data' in st.session_state:
            st.subheader("�� Processed Data Sample")
            st.dataframe(st.session_state['processed_data'].head(), width="stretch")
    
    def feature_engineering_page(self):
        """Display feature engineering page."""
        st.title("⚙️ Feature Engineering")
        
        # Check if data is loaded in session state
        if not st.session_state.get('data_loaded', False) or self.data is None:
            st.warning("Please load data first in the Data Explorer.")
            return
        
        # Initialize feature engineer
        if self.feature_engineer is None:
            self.feature_engineer = FeatureEngineer()
        
        st.subheader("��️ Feature Engineering Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            time_features = st.checkbox("Create Time Features", value=True)
            amount_features = st.checkbox("Create Amount Features", value=True)
            balance_features = st.checkbox("Create Balance Features", value=True)
        
        with col2:
            transaction_features = st.checkbox("Create Transaction Type Features", value=True)
            fraud_indicators = st.checkbox("Create Fraud Indicators", value=True)
        
        if st.button("Engineer Features"):
            with st.spinner("Engineering features..."):
                try:
                    # Use processed data if available, otherwise use original data
                    input_data = st.session_state.get('processed_data', self.data)
                    engineered_data = input_data.copy()
                    
                    if time_features:
                        engineered_data = self.feature_engineer.create_time_features(engineered_data)
                        st.success("✅ Time features created")
                    
                    if amount_features:
                        engineered_data = self.feature_engineer.create_amount_features(engineered_data)
                        st.success("✅ Amount features created")
                    
                    if balance_features:
                        engineered_data = self.feature_engineer.create_balance_features(engineered_data)
                        st.success("✅ Balance features created")
                    
                    if transaction_features:
                        engineered_data = self.feature_engineer.create_transaction_type_features(engineered_data)
                        st.success("✅ Transaction type features created")
                    
                    if fraud_indicators:
                        engineered_data = self.feature_engineer.create_fraud_indicators(engineered_data)
                        st.success("✅ Fraud indicator features created")
                    
                    # Store engineered data
                    st.session_state['engineered_data'] = engineered_data
                    
                    st.success("�� Feature engineering completed successfully!")
                    
                    # Show feature count comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Original Features:**")
                        st.write(f"Count: {input_data.shape[1]}")
                    with col2:
                        st.write("**Engineered Features:**")
                        st.write(f"Count: {engineered_data.shape[1]}")
                        st.write(f"New features: {engineered_data.shape[1] - input_data.shape[1]}")
                    
                except Exception as e:
                    st.error(f"Error during feature engineering: {str(e)}")
        
        # Display new features
        if 'engineered_data' in st.session_state:
            st.subheader("�� New Features")
            input_data = st.session_state.get('processed_data', self.data)
            new_features = [col for col in st.session_state['engineered_data'].columns 
                          if col not in input_data.columns]
            
            if new_features:
                st.write("**Newly created features:**")
                for i, feature in enumerate(new_features, 1):
                    st.write(f"{i}. {feature}")
            
            st.subheader("�� Engineered Data Sample")
            st.dataframe(st.session_state['engineered_data'].head(), width="stretch")
    
    def model_training_page(self):
        """Display model training page."""
        st.title("�� Model Training")
        
        if 'engineered_data' not in st.session_state:
            st.warning("Please complete data preprocessing and feature engineering first.")
            return
        
        # Initialize model trainer
        if self.model_trainer is None:
            self.model_trainer = FraudDetectionModels(random_state=42)
        
        st.subheader("�� Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_column = st.selectbox("Target Column", ['isFraud'], index=0)
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
            cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        
        with col2:
            resample_method = st.selectbox("Class Imbalance Handling", 
                                         ['smote', 'undersample', 'none'])
            models_to_train = st.multiselect(
                "Models to Train",
                ['logistic_regression', 'random_forest', 'gradient_boosting', 
                 'svm', 'naive_bayes', 'neural_network'],
                default=['logistic_regression', 'random_forest', 'gradient_boosting']
            )
        
        if st.button("Train Models"):
            if not models_to_train:
                st.error("Please select at least one model to train.")
                return
            
            with st.spinner("Training models... This may take a few minutes."):
                try:
                    data = st.session_state['engineered_data']
                    
                    # Create preprocessor instance for data splitting
                    temp_preprocessor = DataPreprocessor()
                    
                    # Prepare data
                    X_train, X_test, y_train, y_test = temp_preprocessor.prepare_data(
                        data, target_column, test_size, random_state=42
                    )
                    
                    # Clean data: handle infinite values and very large numbers
                    def clean_data_for_training(X):
                        """Clean data to remove infinite values and very large numbers."""
                        X_clean = X.copy()
                        
                        # Replace infinite values with NaN
                        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
                        
                        # Cap very large values
                        for col in X_clean.select_dtypes(include=[np.number]).columns:
                            # Cap at 99.9th percentile to handle extreme outliers
                            upper_cap = X_clean[col].quantile(0.999)
                            lower_cap = X_clean[col].quantile(0.001)
                            X_clean[col] = X_clean[col].clip(lower=lower_cap, upper=upper_cap)
                        
                        # Fill any remaining NaN values with median
                        X_clean = X_clean.fillna(X_clean.median())
                        
                        return X_clean
                    
                    # Apply cleaning to training and test sets
                    X_train_clean = clean_data_for_training(X_train)
                    X_test_clean = clean_data_for_training(X_test)
                    
                    # Validate cleaned data
                    def validate_data(X, name):
                        """Validate data for training."""
                        issues = []
                        
                        # Check for infinite values
                        if np.isinf(X).any().any():
                            issues.append(f"Infinite values found in {name}")
                        
                        # Check for NaN values
                        if X.isnull().any().any():
                            issues.append(f"NaN values found in {name}")
                        
                        # Check for very large values
                        if (np.abs(X.select_dtypes(include=[np.number])) > 1e15).any().any():
                            issues.append(f"Very large values found in {name}")
                        
                        return issues
                    
                    # Validate both sets
                    train_issues = validate_data(X_train_clean, "training data")
                    test_issues = validate_data(X_test_clean, "test data")
                    
                    if train_issues or test_issues:
                        st.warning("Data validation issues found:")
                        for issue in train_issues + test_issues:
                            st.write(f"- {issue}")
                        st.write("Attempting additional cleaning...")
                        
                        # Additional aggressive cleaning
                        X_train_clean = X_train_clean.fillna(0)
                        X_test_clean = X_test_clean.fillna(0)
                        X_train_clean = X_train_clean.replace([np.inf, -np.inf], 0)
                        X_test_clean = X_test_clean.replace([np.inf, -np.inf], 0)
                    
                    st.info(f"✅ Data cleaned and validated. Training shape: {X_train_clean.shape}, Test shape: {X_test_clean.shape}")
                    
                    # Train selected models
                    results = {}
                    progress_bar = st.progress(0)
                    
                    for i, model_name in enumerate(models_to_train):
                        st.write(f"Training {model_name}...")
                        
                        # Get the specific model
                        base_models = self.model_trainer.get_base_models()
                        if model_name in base_models:
                            model_results = self.model_trainer.train_single_model(
                                model_name, base_models[model_name], X_train_clean, y_train, cv_folds
                            )
                            results[model_name] = model_results
                        
                        progress_bar.progress((i + 1) / len(models_to_train))
                    
                    # Store results and data splits (use cleaned data)
                    st.session_state['training_results'] = results
                    st.session_state['data_splits'] = {
                        'X_train': X_train_clean, 'X_test': X_test_clean,
                        'y_train': y_train, 'y_test': y_test
                    }
                    st.session_state['trained_models'] = self.model_trainer.models
                    
                    st.success("�� Model training completed!")
                    
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
                    
                    # Provide diagnostic information
                    st.write("**Diagnostic Information:**")
                    try:
                        data = st.session_state['engineered_data']
                        st.write(f"- Data shape: {data.shape}")
                        st.write(f"- Data types: {data.dtypes.value_counts().to_dict()}")
                        
                        # Check for problematic values
                        numeric_cols = data.select_dtypes(include=[np.number]).columns
                        inf_cols = []
                        large_cols = []
                        
                        for col in numeric_cols:
                            if np.isinf(data[col]).any():
                                inf_cols.append(col)
                            if (np.abs(data[col]) > 1e10).any():
                                large_cols.append(col)
                        
                        if inf_cols:
                            st.write(f"- Columns with infinite values: {inf_cols}")
                        if large_cols:
                            st.write(f"- Columns with very large values: {large_cols}")
                            
                    except Exception as diag_e:
                        st.write(f"Could not generate diagnostics: {str(diag_e)}")
                    
                    st.write("**Troubleshooting:**")
                    st.write("- Try using a smaller sample of data first")
                    st.write("- Check the Data Explorer for any unusual values")
                    st.write("- Consider different preprocessing options")
        
        # Display training results
        if 'training_results' in st.session_state:
            st.subheader("�� Training Results")
            
            results_df = pd.DataFrame(st.session_state['training_results']).T
            results_df = results_df.sort_values('cv_mean_auc', ascending=False)
            
            st.dataframe(results_df.style.highlight_max(axis=0), width="stretch")
            
            # Plot model comparison
            fig = px.bar(
                x=results_df.index,
                y=results_df['cv_mean_auc'],
                error_y=results_df['cv_std_auc'],
                title="Model Performance Comparison (Cross-Validation AUC)",
                labels={'x': 'Model', 'y': 'CV AUC Score'}
            )
            st.plotly_chart(fig, width="stretch")
    
    def evaluation_page(self):
        """Display model evaluation page."""
        st.title("�� Model Evaluation")
        
        if 'trained_models' not in st.session_state:
            st.warning("Please train models first.")
            return
        
        # Initialize evaluator
        if self.evaluator is None:
            self.evaluator = ModelEvaluator()
        
        st.subheader("�� Model Selection")
        
        model_names = list(st.session_state['trained_models'].keys())
        selected_model = st.selectbox("Select Model for Evaluation", model_names)
        
        if st.button("Evaluate Model"):
            with st.spinner("Evaluating model..."):
                try:
                    # Get data splits and model
                    data_splits = st.session_state['data_splits']
                    X_test = data_splits['X_test']
                    y_test = data_splits['y_test']
                    model = st.session_state['trained_models'][selected_model]
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    # Generate evaluation report
                    report = self.evaluator.generate_evaluation_report(
                        y_test.values, y_pred, y_prob, model_name=selected_model
                    )
                    
                    st.session_state['evaluation_report'] = report
                    st.session_state['predictions'] = {
                        'y_test': y_test.values, 'y_pred': y_pred, 'y_prob': y_prob
                    }
                    
                    st.success("✅ Model evaluation completed!")
                    
                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")
        
        # Display evaluation results
        if 'evaluation_report' in st.session_state:
            report = st.session_state['evaluation_report']
            
            st.subheader("�� Performance Metrics")
            
            # Basic metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Classification Metrics:**")
                basic_metrics = report['basic_metrics']
                for metric, value in basic_metrics.items():
                    st.metric(metric.replace('_', ' ').title(), f"{value:.4f}")
            
            with col2:
                st.write("**Business Metrics:**")
                business_metrics = report['business_metrics']
                for metric, value in business_metrics.items():
                    if isinstance(value, (int, float)):
                        st.metric(metric.replace('_', ' ').title(), f"{value:.4f}")
            
            # Visualizations
            if 'predictions' in st.session_state:
                pred_data = st.session_state['predictions']
                
                st.subheader("�� Performance Visualizations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # ROC Curve
                    if pred_data['y_prob'] is not None:
                        fig_roc = self.evaluator.plot_roc_curve(
                            pred_data['y_test'], pred_data['y_prob'], selected_model
                        )
                        st.pyplot(fig_roc)
                
                with col2:
                    # Confusion Matrix
                    fig_cm = self.evaluator.plot_confusion_matrix(
                        pred_data['y_test'], pred_data['y_pred']
                    )
                    st.pyplot(fig_cm)
    
    def prediction_page(self):
        """Display prediction page for new transactions."""
        st.title("�� Transaction Prediction")
        
        if 'trained_models' not in st.session_state:
            st.warning("Please train models first.")
            return
        
        st.subheader("�� Enter Transaction Details")
        
        # Model selection
        model_names = list(st.session_state['trained_models'].keys())
        selected_model = st.selectbox("Select Model", model_names)
        
        col1, col2 = st.columns(2)
        
        with col1:
            step = st.number_input("Step (Hour)", min_value=1, max_value=8760, value=100)
            transaction_type = st.selectbox("Transaction Type", 
                                          ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'])
            amount = st.number_input("Amount", min_value=0.0, value=1000.0, step=0.01)
            old_balance_orig = st.number_input("Origin Old Balance", min_value=0.0, value=5000.0)
            new_balance_orig = st.number_input("Origin New Balance", min_value=0.0, value=4000.0)
        
        with col2:
            old_balance_dest = st.number_input("Destination Old Balance", min_value=0.0, value=0.0)
            new_balance_dest = st.number_input("Destination New Balance", min_value=0.0, value=1000.0)
        
        if st.button("Predict Fraud Risk"):
            try:
                # Create transaction dataframe
                transaction_data = pd.DataFrame({
                    'step': [step],
                    'type': [transaction_type],
                    'amount': [amount],
                    'oldbalanceOrg': [old_balance_orig],
                    'newbalanceOrig': [new_balance_orig],
                    'oldbalanceDest': [old_balance_dest],
                    'newbalanceDest': [new_balance_dest],
                    'nameOrig': ['C_TEST'],
                    'nameDest': ['M_TEST']
                })
                
                # Apply same preprocessing and feature engineering
                if self.preprocessor is None:
                    self.preprocessor = DataPreprocessor()
                if self.feature_engineer is None:
                    self.feature_engineer = FeatureEngineer()
                
                # Preprocess
                processed_transaction = self.preprocessor.encode_categorical_features(transaction_data)
                
                # Engineer features
                engineered_transaction = self.feature_engineer.engineer_features(processed_transaction)
                
                # Get model and make prediction
                model = st.session_state['trained_models'][selected_model]
                
                # Ensure all required columns are present
                # This is a simplified version - in production, you'd need proper feature alignment
                required_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                                   'oldbalanceDest', 'newbalanceDest']
                
                # Create a minimal feature set for prediction
                feature_data = engineered_transaction[
                    [col for col in required_features if col in engineered_transaction.columns]
                ].iloc[0:1]
                
                # Fill missing features with zeros (this is simplified)
                for col in required_features:
                    if col not in feature_data.columns:
                        feature_data[col] = 0
                
                prediction = model.predict(feature_data)[0]
                probability = model.predict_proba(feature_data)[0] if hasattr(model, 'predict_proba') else None
                
                # Display results
                st.subheader("�� Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.error("�� HIGH FRAUD RISK")
                        st.write("This transaction shows characteristics of potential fraud.")
                    else:
                        st.success("✅ LOW FRAUD RISK")
                        st.write("This transaction appears to be legitimate.")
                
                with col2:
                    if probability is not None:
                        fraud_prob = probability[1] * 100
                        st.metric("Fraud Probability", f"{fraud_prob:.2f}%")
                        
                        # Risk gauge
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = fraud_prob,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Fraud Risk %"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 25], 'color': "lightgray"},
                                    {'range': [25, 50], 'color': "yellow"},
                                    {'range': [50, 75], 'color': "orange"},
                                    {'range': [75, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        st.plotly_chart(fig_gauge, width="stretch")
                
                # Risk factors
                st.subheader("�� Risk Factors Analysis")
                risk_factors = []
                
                if transaction_type in ['TRANSFER', 'CASH_OUT']:
                    risk_factors.append("High-risk transaction type")
                if amount > 10000:
                    risk_factors.append("High transaction amount")
                if old_balance_orig == 0 and new_balance_orig == 0:
                    risk_factors.append("Suspicious balance pattern")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.warning(f"⚠️ {factor}")
                else:
                    st.info("ℹ️ No major risk factors detected")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    def run(self):
        """Run the Streamlit application."""
        # Sidebar navigation
        page = self.sidebar_navigation()
        
        # Add data loading info in sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("�� Data Status")
        
        # Large dataset help
        with st.sidebar.expander("�� Large Dataset Help"):
            st.write("**Can't upload 470MB file?**")
            st.write("• Use 'Local File Path' tab")
            st.write("• Place file in data/raw/ folder")
            st.write("• Start with sample data")
            st.write("• See LARGE_DATA_GUIDE.md")
        if st.session_state.get('data_loaded', False) and self.data is not None:
            st.sidebar.success(f"Data loaded: {self.data.shape[0]} rows")
        else:
            st.sidebar.info("No data loaded")
        
        if 'processed_data' in st.session_state:
            st.sidebar.success("Data preprocessed ✅")
        
        if 'engineered_data' in st.session_state:
            st.sidebar.success("Features engineered ✅")
        
        if 'trained_models' in st.session_state:
            st.sidebar.success("Models trained ✅")
        
        # Page routing
        if page == "home":
            self.home_page()
        elif page == "data_explorer":
            self.data_explorer_page()
        elif page == "preprocessing":
            self.preprocessing_page()
        elif page == "feature_engineering":
            self.feature_engineering_page()
        elif page == "model_training":
            self.model_training_page()
        elif page == "evaluation":
            self.evaluation_page()
        elif page == "prediction":
            self.prediction_page()


def main():
    """Main function to run the Streamlit app."""
    app = FraudDetectionApp()
    app.run()


if __name__ == "__main__":
    main()