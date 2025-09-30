"""
Data Preprocessing Utilities for ML Framework
============================================

This module contains comprehensive data preprocessing functions with detailed explanations
of how each function works and why it's needed in the machine learning pipeline.

Author: ML Framework Team
Last Updated: September 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessor:
    """
    A comprehensive data preprocessing class that handles all common preprocessing tasks.
    
    How it works:
    - Stores preprocessing parameters for consistent application to train/test data
    - Provides methods for each preprocessing step with detailed explanations
    - Maintains state to ensure test data is processed identically to training data
    """
    
    def __init__(self):
        """
        Initialize the preprocessor.
        
        What this does:
        - Creates empty containers for storing preprocessing objects (scalers, encoders, etc.)
        - These objects will be fitted on training data and applied to both train and test data
        
        Why we need this:
        - Prevents data leakage by ensuring test data preprocessing uses only training data statistics
        - Maintains consistency across different datasets
        """
        self.scaler = None
        self.label_encoders = {}
        self.imputers = {}
        self.feature_names = None
        
    def basic_info(self, df, title="Dataset Information"):
        """
        Display comprehensive information about a dataset.
        
        What this does:
        - Shows dataset shape, data types, missing values, and basic statistics
        - Provides a complete overview of data quality and structure
        
        Parameters:
        - df: pandas DataFrame to analyze
        - title: string title for the analysis
        
        How it works:
        1. Uses pandas methods to extract dataset metadata
        2. Calculates missing value percentages
        3. Displays statistical summaries for numerical columns
        """
        print(f"\n{'='*50}")
        print(f"{title}")
        print(f"{'='*50}")
        
        # Basic dataset information
        print(f"📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"💾 Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data types analysis
        print(f"\n📋 Data Types:")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   {dtype}: {count} columns")
        
        # Missing values analysis
        print(f"\n❌ Missing Values:")
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Missing Percentage': missing_percent
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing_df) > 0:
            print(missing_df)
        else:
            print("   No missing values found! ✅")
        
        # Numerical columns summary
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            print(f"\n📈 Numerical Columns Summary:")
            print(df[numerical_cols].describe())
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"\n📝 Categorical Columns Summary:")
            for col in categorical_cols:
                unique_count = df[col].nunique()
                print(f"   {col}: {unique_count} unique values")
                if unique_count <= 10:  # Show values if not too many
                    print(f"      Values: {list(df[col].unique())}")
    
    def handle_missing_values(self, df, strategy='mean', columns=None):
        """
        Handle missing values in the dataset.
        
        What this does:
        - Fills missing values using specified strategy
        - Stores imputer for consistent application to test data
        
        Parameters:
        - df: DataFrame to process
        - strategy: 'mean', 'median', 'most_frequent', or 'constant'
        - columns: list of columns to process (None = all columns with missing values)
        
        How it works:
        1. Identifies columns with missing values
        2. Creates and fits SimpleImputer with chosen strategy
        3. Transforms the data and returns clean DataFrame
        
        Why each strategy:
        - 'mean': Good for normally distributed numerical data
        - 'median': Better for skewed numerical data (robust to outliers)
        - 'most_frequent': Good for categorical data
        - 'constant': When you want to fill with a specific value
        """
        print(f"\n🔧 Handling Missing Values with '{strategy}' strategy...")
        
        if columns is None:
            # Find columns with missing values
            columns = df.columns[df.isnull().any()].tolist()
        
        if not columns:
            print("   No missing values to handle! ✅")
            return df.copy()
        
        df_processed = df.copy()
        
        for col in columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                print(f"   Processing {col}: {missing_count} missing values")
                
                # Create imputer for this column
                if col not in self.imputers:
                    self.imputers[col] = SimpleImputer(strategy=strategy)
                    # Fit on the current data (should be training data)
                    self.imputers[col].fit(df[[col]])
                
                # Transform the data
                df_processed[col] = self.imputers[col].transform(df[[col]]).flatten()
        
        print(f"   ✅ Missing values handled for {len(columns)} columns")
        return df_processed
    
    def encode_categorical(self, df, columns=None, method='label'):
        """
        Encode categorical variables to numerical format.
        
        What this does:
        - Converts categorical text data to numbers that ML algorithms can use
        - Stores encoders for consistent application to test data
        
        Parameters:
        - df: DataFrame to process
        - columns: list of categorical columns (None = auto-detect)
        - method: 'label' for label encoding, 'onehot' for one-hot encoding
        
        How Label Encoding works:
        - Assigns a unique number to each category
        - Example: ['red', 'blue', 'green'] → [0, 1, 2]
        - Good for ordinal data or when categories have natural order
        
        How One-Hot Encoding works:
        - Creates binary columns for each category
        - Example: 'color' → 'color_red', 'color_blue', 'color_green' (0/1 values)
        - Good for nominal data where categories have no natural order
        """
        print(f"\n🏷️ Encoding Categorical Variables using '{method}' method...")
        
        if columns is None:
            # Auto-detect categorical columns
            columns = df.select_dtypes(include=['object']).columns.tolist()
        
        if not columns:
            print("   No categorical columns to encode! ✅")
            return df.copy()
        
        df_processed = df.copy()
        
        for col in columns:
            unique_values = df[col].nunique()
            print(f"   Processing {col}: {unique_values} unique categories")
            
            if method == 'label':
                # Label Encoding: each category gets a number
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    self.label_encoders[col].fit(df[col])
                
                df_processed[col] = self.label_encoders[col].transform(df[col])
                
            elif method == 'onehot':
                # One-Hot Encoding: each category gets its own column
                if unique_values <= 10:  # Only for reasonable number of categories
                    dummies = pd.get_dummies(df[col], prefix=col)
                    df_processed = pd.concat([df_processed.drop(col, axis=1), dummies], axis=1)
                    print(f"     Created {len(dummies.columns)} dummy columns")
                else:
                    print(f"     Skipping {col}: too many categories ({unique_values})")
        
        print(f"   ✅ Categorical encoding completed")
        return df_processed
    
    def scale_features(self, df, method='standard', columns=None):
        """
        Scale numerical features to similar ranges.
        
        What this does:
        - Transforms numerical features to have similar scales
        - Prevents features with large values from dominating the model
        
        Parameters:
        - df: DataFrame to process
        - method: 'standard' or 'minmax'
        - columns: list of columns to scale (None = all numerical columns)
        
        How Standard Scaling works:
        - Formula: (x - mean) / standard_deviation
        - Result: mean = 0, standard deviation = 1
        - Good for normally distributed data
        - Example: [1, 2, 3, 4, 5] → [-1.41, -0.71, 0, 0.71, 1.41]
        
        How MinMax Scaling works:
        - Formula: (x - min) / (max - min)
        - Result: values between 0 and 1
        - Good for bounded data or when you need specific range
        - Example: [1, 2, 3, 4, 5] → [0, 0.25, 0.5, 0.75, 1.0]
        """
        print(f"\n📏 Scaling Features using '{method}' method...")
        
        if columns is None:
            # Auto-detect numerical columns
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not columns:
            print("   No numerical columns to scale! ⚠️")
            return df.copy()
        
        df_processed = df.copy()
        
        # Create scaler if not exists
        if self.scaler is None:
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
            
            # Fit on the current data (should be training data)
            self.scaler.fit(df[columns])
        
        # Transform the data
        df_processed[columns] = self.scaler.transform(df[columns])
        
        print(f"   ✅ Scaled {len(columns)} numerical columns")
        print(f"   Columns: {columns}")
        
        return df_processed
    
    def create_train_test_split(self, df, target_column, test_size=0.2, random_state=42):
        """
        Split dataset into training and testing sets.
        
        What this does:
        - Separates features (X) and target variable (y)
        - Randomly splits data into training and testing portions
        - Ensures balanced representation in both sets
        
        Parameters:
        - df: DataFrame containing features and target
        - target_column: name of the target variable column
        - test_size: proportion of data for testing (0.2 = 20%)
        - random_state: seed for reproducible results
        
        How it works:
        1. Separates features (X) from target (y)
        2. Uses sklearn's train_test_split for random sampling
        3. Stratification ensures balanced class distribution (for classification)
        
        Why we split data:
        - Training set: Used to train the model (learn patterns)
        - Testing set: Used to evaluate model performance (unseen data)
        - Prevents overfitting by testing on data the model hasn't seen
        """
        print(f"\n✂️ Splitting Data into Train/Test Sets...")
        print(f"   Test size: {test_size*100}% | Random state: {random_state}")
        
        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        print(f"   Features shape: {X.shape}")
        print(f"   Target shape: {y.shape}")
        
        # Determine if classification or regression
        is_classification = len(y.unique()) < 20  # Heuristic
        
        if is_classification:
            # For classification, use stratification to maintain class balance
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            print(f"   Classification detected - using stratified split")
            print(f"   Class distribution in training set:")
            print(y_train.value_counts(normalize=True))
        else:
            # For regression, simple random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            print(f"   Regression detected - using simple random split")
        
        print(f"   ✅ Training set: {X_train.shape[0]} samples")
        print(f"   ✅ Testing set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def full_preprocessing_pipeline(self, df, target_column, test_size=0.2):
        """
        Complete preprocessing pipeline that applies all steps in correct order.
        
        What this does:
        - Applies all preprocessing steps in the optimal sequence
        - Returns clean, model-ready training and testing data
        
        Pipeline order (IMPORTANT):
        1. Handle missing values (before any other processing)
        2. Encode categorical variables (before scaling)
        3. Split data (before fitting scalers to avoid data leakage)
        4. Scale features (fit on training data only)
        
        Why this order matters:
        - Missing values first: Other operations might fail with NaN values
        - Encoding before splitting: Ensures consistent categories
        - Scaling after splitting: Prevents test data information leaking into training
        """
        print(f"\n🔄 Running Full Preprocessing Pipeline...")
        print(f"{'='*60}")
        
        # Step 1: Basic info about raw data
        self.basic_info(df, "Raw Dataset Analysis")
        
        # Step 2: Handle missing values
        df_clean = self.handle_missing_values(df)
        
        # Step 3: Encode categorical variables
        df_encoded = self.encode_categorical(df_clean)
        
        # Step 4: Split the data
        X_train, X_test, y_train, y_test = self.create_train_test_split(
            df_encoded, target_column, test_size
        )
        
        # Step 5: Scale features (fit on training data only)
        X_train_scaled = self.scale_features(X_train)
        X_test_scaled = self.scale_features(X_test)  # Uses training data statistics
        
        print(f"\n✅ Preprocessing Pipeline Complete!")
        print(f"   Final training set shape: {X_train_scaled.shape}")
        print(f"   Final testing set shape: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test

# Example usage and educational functions
def demonstrate_preprocessing():
    """
    Demonstration function showing how to use the DataPreprocessor class.
    
    This function provides examples and explanations for educational purposes.
    """
    print("DataPreprocessor Demonstration")
    print("="*50)
    
    # Create sample data for demonstration
    sample_data = pd.DataFrame({
        'age': [25, 30, np.nan, 45, 35],
        'income': [50000, 60000, 55000, np.nan, 70000],
        'category': ['A', 'B', 'A', 'C', 'B'],
        'target': [0, 1, 0, 1, 1]
    })
    
    print("Sample dataset:")
    print(sample_data)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Run full pipeline
    X_train, X_test, y_train, y_test = preprocessor.full_preprocessing_pipeline(
        sample_data, 'target', test_size=0.4
    )
    
    print("\nProcessed training features:")
    print(X_train)
    print("\nProcessed testing features:")
    print(X_test)

if __name__ == "__main__":
    # Run demonstration when script is executed directly
    demonstrate_preprocessing()