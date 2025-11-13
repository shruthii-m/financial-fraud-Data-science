"""
Data Loading Utility for Large Datasets

This script helps load the 470MB Kaggle fraud detection dataset
efficiently with various sampling and chunking options.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

def load_fraud_dataset(file_path, 
                      sample_rows=None, 
                      use_chunks=False, 
                      chunk_size=50000,
                      save_sample=True):
    """
    Load the fraud detection dataset with memory optimization.
    
    Args:
        file_path (str): Path to the CSV file
        sample_rows (int): Number of rows to sample (None for all)
        use_chunks (bool): Whether to use chunked reading
        chunk_size (int): Size of chunks when reading
        save_sample (bool): Whether to save a sample for quick access
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    
    print(f"�� Loading dataset from: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get file size
    file_size_mb = os.path.getsize(file_path) / 1024 / 1024
    print(f"�� File size: {file_size_mb:.1f} MB")
    
    start_time = datetime.now()
    
    try:
        if use_chunks and sample_rows is None:
            # Read in chunks for very large files
            print(f"�� Reading in chunks of {chunk_size:,} rows...")
            chunks = []
            total_rows = 0
            
            for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
                chunks.append(chunk)
                total_rows += len(chunk)
                print(f"   Chunk {i+1}: {len(chunk):,} rows (Total: {total_rows:,})")
                
                # Limit memory usage - stop at reasonable size
                if len(chunks) >= 20:  # ~1M rows max
                    print("   Stopping at 1M rows to preserve memory")
                    break
            
            df = pd.concat(chunks, ignore_index=True)
            
        else:
            # Read normally with optional row limit
            if sample_rows:
                print(f"�� Reading first {sample_rows:,} rows...")
                df = pd.read_csv(file_path, nrows=sample_rows)
            else:
                print("�� Reading full dataset...")
                df = pd.read_csv(file_path)
        
        # Calculate loading time and memory usage
        load_time = (datetime.now() - start_time).total_seconds()
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        print(f"✅ Data loaded successfully!")
        print(f"   Shape: {df.shape}")
        print(f"   Memory usage: {memory_mb:.1f} MB")
        print(f"   Loading time: {load_time:.1f} seconds")
        
        # Display basic info
        if 'isFraud' in df.columns:
            fraud_count = df['isFraud'].sum()
            fraud_rate = fraud_count / len(df) * 100
            print(f"   Fraud cases: {fraud_count:,} ({fraud_rate:.2f}%)")
        
        # Save sample for quick access if requested
        if save_sample and len(df) > 10000:
            sample_path = "data/processed/sample_data.csv"
            os.makedirs("data/processed", exist_ok=True)
            
            sample_df = df.sample(n=min(10000, len(df)), random_state=42)
            sample_df.to_csv(sample_path, index=False)
            print(f"�� Saved 10K sample to: {sample_path}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        raise


def create_smaller_samples(df, output_dir="data/processed"):
    """Create multiple sample sizes for different use cases."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    sample_sizes = {
        "sample_1k.csv": 1000,
        "sample_10k.csv": 10000,
        "sample_50k.csv": 50000,
        "sample_100k.csv": 100000
    }
    
    print("\n�� Creating sample files...")
    
    for filename, size in sample_sizes.items():
        if len(df) >= size:
            sample = df.sample(n=size, random_state=42)
            sample_path = os.path.join(output_dir, filename)
            sample.to_csv(sample_path, index=False)
            print(f"   ✅ {filename}: {size:,} rows")
        else:
            print(f"   ⚠️ {filename}: Dataset too small ({len(df):,} rows)")


def main():
    """Main function for command line usage."""
    
    print("�� Financial Fraud Detection - Data Loader")
    print("=" * 50)
    
    # Default file path
    default_path = "data/raw/fraud_detection_dataset.csv"
    
    # Check if default file exists
    if os.path.exists(default_path):
        file_path = default_path
        print(f"�� Found dataset at: {file_path}")
    else:
        print(f"�� Dataset not found at: {default_path}")
        print("\n�� Please:")
        print("1. Download the Kaggle dataset")
        print("2. Place it in: data/raw/fraud_detection_dataset.csv")
        print("3. Or specify the path below")
        
        file_path = input("\nEnter path to your CSV file: ").strip()
        if not file_path:
            print("❌ No file path provided. Exiting.")
            return
    
    # Loading options
    print(f"\n⚙️ Loading Options:")
    print("1. Full dataset (may use lots of memory)")
    print("2. Sample 100K rows (recommended for testing)")
    print("3. Sample 50K rows (fast loading)")
    print("4. Chunked reading (memory efficient)")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    try:
        if choice == "1":
            df = load_fraud_dataset(file_path)
        elif choice == "2":
            df = load_fraud_dataset(file_path, sample_rows=100000)
        elif choice == "3":
            df = load_fraud_dataset(file_path, sample_rows=50000)
        elif choice == "4":
            df = load_fraud_dataset(file_path, use_chunks=True)
        else:
            print("Invalid choice. Using default (100K rows)...")
            df = load_fraud_dataset(file_path, sample_rows=100000)
        
        # Ask about creating samples
        create_samples = input("\nCreate sample files for Streamlit app? (y/n): ").strip().lower()
        if create_samples in ['y', 'yes']:
            create_smaller_samples(df)
        
        print(f"\n�� Data loading complete!")
        print(f"   Dataset ready for analysis: {df.shape}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n�� Troubleshooting:")
        print("- Check file path is correct")
        print("- Ensure file is not open in another program")
        print("- Try with smaller sample size")


if __name__ == "__main__":
    main()