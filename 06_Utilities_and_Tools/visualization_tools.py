"""
Visualization Tools for ML Framework
===================================

This module contains comprehensive visualization functions with detailed explanations
of how each plot works and when to use it in machine learning projects.

Author: ML Framework Team
Last Updated: September 2025
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import validation_curve, learning_curve
import warnings
warnings.filterwarnings('ignore')

# Set plotting style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MLVisualizer:
    """
    A comprehensive visualization class for machine learning projects.
    
    How it works:
    - Provides methods for all common ML visualizations
    - Each method includes detailed explanations of what the plot shows
    - Uses both static (matplotlib/seaborn) and interactive (plotly) visualizations
    """
    
    def __init__(self, figsize=(10, 6)):
        """
        Initialize the visualizer with default settings.
        
        Parameters:
        - figsize: default figure size for matplotlib plots
        """
        self.figsize = figsize
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    def plot_data_distribution(self, df, columns=None, title="Data Distribution Analysis"):
        """
        Visualize the distribution of numerical columns in the dataset.
        
        What this shows:
        - Shape of data distribution (normal, skewed, bimodal, etc.)
        - Presence of outliers
        - Range and spread of values
        
        When to use:
        - During exploratory data analysis (EDA)
        - Before choosing preprocessing methods
        - To understand data quality issues
        
        How to interpret:
        - Bell curve: normally distributed data (good for many algorithms)
        - Skewed: might need log transformation
        - Multiple peaks: might indicate mixed populations
        - Long tails: potential outliers
        """
        print(f"\n📊 {title}")
        print("="*50)
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_cols = len(columns)
        if n_cols == 0:
            print("No numerical columns found!")
            return
        
        # Calculate subplot layout
        n_rows = (n_cols + 2) // 3  # 3 columns per row
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(columns):
            ax = axes[i] if n_cols > 1 else axes
            
            # Create histogram with density curve
            ax.hist(df[col].dropna(), bins=30, alpha=0.7, color=self.colors[i % len(self.colors)], 
                   density=True, label='Histogram')
            
            # Add density curve
            data_clean = df[col].dropna()
            if len(data_clean) > 1:
                try:
                    # Create density curve
                    x_range = np.linspace(data_clean.min(), data_clean.max(), 100)
                    density = np.histogram(data_clean, bins=30, density=True)[0]
                    ax2 = ax.twinx()
                    ax2.plot(data_clean.sort_values(), np.arange(len(data_clean))/len(data_clean), 
                            color='red', linewidth=2, label='Cumulative')
                    ax2.set_ylabel('Cumulative Probability')
                    ax2.legend(loc='upper right')
                except:
                    pass
            
            ax.set_title(f'{col}\nMean: {df[col].mean():.2f}, Std: {df[col].std():.2f}')
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
            ax.legend(loc='upper left')
            
            # Add interpretation text
            skewness = df[col].skew()
            if abs(skewness) < 0.5:
                skew_text = "Normal"
            elif skewness > 0.5:
                skew_text = "Right-skewed"
            else:
                skew_text = "Left-skewed"
            
            ax.text(0.02, 0.98, f'Distribution: {skew_text}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for i in range(n_cols, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistical summary
        print("\n📈 Statistical Summary:")
        print(df[columns].describe())
    
    def plot_correlation_matrix(self, df, title="Feature Correlation Matrix"):
        """
        Display correlation matrix showing relationships between numerical features.
        
        What this shows:
        - Linear relationships between pairs of variables
        - Strength and direction of correlations
        - Potential multicollinearity issues
        
        How to interpret:
        - Values range from -1 to +1
        - +1: Perfect positive correlation (as one increases, other increases)
        - -1: Perfect negative correlation (as one increases, other decreases)
        - 0: No linear relationship
        - |correlation| > 0.8: Strong correlation (potential multicollinearity)
        
        When to use:
        - Feature selection (remove highly correlated features)
        - Understanding feature relationships
        - Detecting potential data quality issues
        """
        print(f"\n🔗 {title}")
        print("="*50)
        
        # Select only numerical columns
        numerical_cols = df.select_dtypes(include=[np.number])
        
        if numerical_cols.empty:
            print("No numerical columns found for correlation analysis!")
            return
        
        # Calculate correlation matrix
        corr_matrix = numerical_cols.corr()
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .5})
        
        plt.title(f'{title}\nInterpretation: Red=Positive, Blue=Negative, White=No correlation')
        plt.tight_layout()
        plt.show()
        
        # Find and report high correlations
        print("\n🎯 High Correlations (|correlation| > 0.7):")
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_val
                    })
        
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs)
            print(high_corr_df.to_string(index=False))
            print("\n💡 Consider removing one feature from highly correlated pairs to reduce multicollinearity.")
        else:
            print("No high correlations found. Good for model stability! ✅")
    
    def plot_target_distribution(self, y, title="Target Variable Distribution"):
        """
        Visualize the distribution of the target variable.
        
        What this shows:
        - Class balance (for classification)
        - Target value distribution (for regression)
        - Potential data imbalance issues
        
        How to interpret:
        Classification:
        - Balanced: roughly equal class sizes (good)
        - Imbalanced: one class dominates (might need special handling)
        
        Regression:
        - Normal distribution: good for most algorithms
        - Skewed: might need transformation
        - Outliers: might affect model performance
        
        When to use:
        - Before model training to understand target characteristics
        - To decide on evaluation metrics
        - To determine if special techniques are needed (e.g., class weighting)
        """
        print(f"\n🎯 {title}")
        print("="*50)
        
        # Determine if classification or regression
        unique_values = len(y.unique())
        is_classification = unique_values < 20  # Heuristic
        
        if is_classification:
            # Classification: Bar plot with counts and percentages
            plt.figure(figsize=self.figsize)
            
            value_counts = y.value_counts()
            percentages = (value_counts / len(y)) * 100
            
            ax = value_counts.plot(kind='bar', color=self.colors[:len(value_counts)])
            plt.title(f'{title}\nTotal samples: {len(y)}')
            plt.xlabel('Classes')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
            # Add percentage labels on bars
            for i, (count, pct) in enumerate(zip(value_counts.values, percentages.values)):
                ax.text(i, count + max(value_counts) * 0.01, f'{pct:.1f}%', 
                       ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
            
            print(f"\n📊 Class Distribution:")
            for class_val, count in value_counts.items():
                percentage = (count / len(y)) * 100
                print(f"   Class {class_val}: {count} samples ({percentage:.1f}%)")
            
            # Check for imbalance
            min_class_pct = percentages.min()
            max_class_pct = percentages.max()
            imbalance_ratio = max_class_pct / min_class_pct
            
            if imbalance_ratio > 3:
                print(f"\n⚠️ Class Imbalance Detected! Ratio: {imbalance_ratio:.1f}:1")
                print("   Consider techniques like:")
                print("   - Class weighting (class_weight='balanced')")
                print("   - SMOTE for oversampling minority class")
                print("   - Stratified sampling")
            else:
                print(f"\n✅ Classes are reasonably balanced (ratio: {imbalance_ratio:.1f}:1)")
        
        else:
            # Regression: Histogram with statistics
            plt.figure(figsize=self.figsize)
            
            plt.hist(y, bins=30, alpha=0.7, color=self.colors[0], edgecolor='black')
            plt.axvline(y.mean(), color='red', linestyle='--', label=f'Mean: {y.mean():.2f}')
            plt.axvline(y.median(), color='green', linestyle='--', label=f'Median: {y.median():.2f}')
            
            plt.title(f'{title}\nDistribution Analysis')
            plt.xlabel('Target Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            print(f"\n📊 Target Variable Statistics:")
            print(f"   Mean: {y.mean():.4f}")
            print(f"   Median: {y.median():.4f}")
            print(f"   Std Dev: {y.std():.4f}")
            print(f"   Min: {y.min():.4f}")
            print(f"   Max: {y.max():.4f}")
            print(f"   Skewness: {y.skew():.4f}")
            
            # Interpretation
            skewness = y.skew()
            if abs(skewness) < 0.5:
                print("   📈 Distribution: Approximately normal")
            elif skewness > 0.5:
                print("   📈 Distribution: Right-skewed (consider log transformation)")
            else:
                print("   📈 Distribution: Left-skewed (consider square transformation)")
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, title="Confusion Matrix"):
        """
        Create an enhanced confusion matrix visualization.
        
        What this shows:
        - True Positives (TP): Correctly predicted positive cases
        - True Negatives (TN): Correctly predicted negative cases  
        - False Positives (FP): Incorrectly predicted as positive (Type I error)
        - False Negatives (FN): Incorrectly predicted as negative (Type II error)
        
        How to interpret:
        - Diagonal cells: Correct predictions
        - Off-diagonal cells: Misclassifications
        - Darker colors: Higher values
        - Perfect model: Only diagonal cells would be dark
        
        When to use:
        - After training a classification model
        - To understand which classes are confused with each other
        - To calculate precision, recall, and F1-score manually
        """
        print(f"\n🎯 {title}")
        print("="*50)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        
        # Create the plot
        plt.figure(figsize=(8, 6))
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        
        plt.title(f'{title}\nAccuracy: {(cm.diagonal().sum() / cm.sum()):.3f}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
        
        # Create percentage version
        plt.figure(figsize=(8, 6))
        
        # Annotation with both count and percentage
        annotations = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations.append(f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)')
        
        annotations = np.array(annotations).reshape(cm.shape)
        
        sns.heatmap(cm_percent, annot=annotations, fmt='', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Percentage'})
        
        plt.title(f'{title} (Percentages)\nEach row sums to 100%')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
        
        # Print detailed classification report
        print(f"\n📊 Detailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Explain key metrics
        print(f"\n💡 Metric Explanations:")
        print(f"   Precision: Of all positive predictions, how many were correct?")
        print(f"   Recall: Of all actual positives, how many did we find?") 
        print(f"   F1-Score: Harmonic mean of precision and recall")
        print(f"   Support: Number of actual samples in each class")
    
    def plot_learning_curves(self, estimator, X, y, title="Learning Curves", cv=5):
        """
        Plot learning curves to diagnose bias and variance in the model.
        
        What this shows:
        - Training score vs. dataset size
        - Validation score vs. dataset size
        - Gap between training and validation performance
        
        How to interpret:
        - High bias (underfitting): Both curves plateau at low performance
        - High variance (overfitting): Large gap between training and validation curves
        - Good fit: Curves converge at high performance with small gap
        - More data needed: Validation curve still improving at the end
        
        When to use:
        - To diagnose overfitting/underfitting
        - To determine if more data would help
        - To choose optimal model complexity
        """
        print(f"\n📈 {title}")
        print("="*50)
        
        # Calculate learning curves
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy' if len(np.unique(y)) < 20 else 'r2'
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Create the plot
        plt.figure(figsize=self.figsize)
        
        # Plot training scores
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.2, color='blue')
        
        # Plot validation scores
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.2, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title(f'{title}\nDiagnosing Model Performance vs. Dataset Size')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Interpretation
        final_gap = train_mean[-1] - val_mean[-1]
        final_val_score = val_mean[-1]
        
        print(f"\n🔍 Learning Curve Analysis:")
        print(f"   Final Training Score: {train_mean[-1]:.4f} ± {train_std[-1]:.4f}")
        print(f"   Final Validation Score: {val_mean[-1]:.4f} ± {val_std[-1]:.4f}")
        print(f"   Performance Gap: {final_gap:.4f}")
        
        if final_gap > 0.1:
            print(f"   🚨 Large gap suggests OVERFITTING")
            print(f"      Recommendations:")
            print(f"      - Reduce model complexity")
            print(f"      - Add regularization")
            print(f"      - Get more training data")
        elif final_val_score < 0.7:
            print(f"   🚨 Low performance suggests UNDERFITTING")
            print(f"      Recommendations:")
            print(f"      - Increase model complexity")
            print(f"      - Add more features")
            print(f"      - Try different algorithm")
        else:
            print(f"   ✅ Good balance between bias and variance")
    
    def plot_feature_importance(self, model, feature_names, title="Feature Importance"):
        """
        Visualize feature importance from tree-based models.
        
        What this shows:
        - Which features contribute most to model predictions
        - Relative importance of each feature
        - Features that might be redundant (very low importance)
        
        How to interpret:
        - Higher bars: More important features
        - Features with zero importance: Potentially removable
        - Top features: Focus on data quality for these
        
        When to use:
        - After training tree-based models (Random Forest, XGBoost, etc.)
        - For feature selection
        - To understand model behavior
        - To explain results to stakeholders
        """
        print(f"\n🌟 {title}")
        print("="*50)
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])  # For linear models
        else:
            print("Model doesn't support feature importance!")
            return
        
        # Create DataFrame for easier handling
        feature_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        # Create the plot
        plt.figure(figsize=(10, max(6, len(feature_names) * 0.3)))
        
        bars = plt.barh(range(len(feature_df)), feature_df['importance'], 
                       color=self.colors[0], alpha=0.7)
        
        plt.yticks(range(len(feature_df)), feature_df['feature'])
        plt.xlabel('Importance Score')
        plt.title(f'{title}\nHigher values indicate more important features')
        
        # Add value labels on bars
        for i, (importance, feature) in enumerate(zip(feature_df['importance'], feature_df['feature'])):
            plt.text(importance + max(feature_df['importance']) * 0.01, i, 
                    f'{importance:.4f}', va='center')
        
        plt.tight_layout()
        plt.show()
        
        # Print top features
        top_features = feature_df.tail(10)  # Top 10
        print(f"\n🏆 Top 10 Most Important Features:")
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            print(f"   {i:2d}. {row['feature']:<20} : {row['importance']:.6f}")
        
        # Suggest feature selection threshold
        cumsum_importance = feature_df.sort_values('importance', ascending=False)['importance'].cumsum()
        threshold_90 = cumsum_importance[cumsum_importance <= 0.9 * cumsum_importance.iloc[-1]]
        n_features_90 = len(threshold_90)
        
        print(f"\n💡 Feature Selection Suggestion:")
        print(f"   Top {n_features_90} features explain 90% of total importance")
        print(f"   Consider using only these features to reduce complexity")

# Utility functions for common plotting tasks
def quick_eda(df, target_column=None):
    """
    Quick Exploratory Data Analysis with comprehensive visualizations.
    
    What this does:
    - Provides a complete overview of your dataset
    - Creates all essential plots for understanding your data
    - Gives actionable insights for next steps
    
    When to use:
    - At the beginning of any ML project
    - When exploring new datasets
    - Before deciding on preprocessing steps
    """
    print("🚀 Quick Exploratory Data Analysis")
    print("="*60)
    
    viz = MLVisualizer()
    
    # Basic info
    print(f"\n📋 Dataset Overview:")
    print(f"   Shape: {df.shape}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data distribution
    viz.plot_data_distribution(df)
    
    # Correlation matrix
    viz.plot_correlation_matrix(df)
    
    # Target distribution if specified
    if target_column and target_column in df.columns:
        viz.plot_target_distribution(df[target_column])
    
    print("\n✅ EDA Complete! Use insights to guide preprocessing and modeling decisions.")

if __name__ == "__main__":
    # Example usage
    print("MLVisualizer Demonstration")
    print("="*50)
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.exponential(2, 1000),
        'feature3': np.random.uniform(-1, 1, 1000),
        'target': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
    })
    
    # Run quick EDA
    quick_eda(sample_data, 'target')