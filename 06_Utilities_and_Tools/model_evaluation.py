"""
Model Evaluation Utilities for ML Framework
==========================================

This module contains comprehensive model evaluation functions with detailed explanations
of metrics, when to use them, and how to interpret results.

Author: ML Framework Team
Last Updated: September 2025
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import Dict, List, Tuple, Any

class ModelEvaluator:
    """
    Comprehensive model evaluation class with detailed explanations.
    
    How it works:
    - Provides evaluation methods for both classification and regression
    - Explains what each metric means and when to use it
    - Compares multiple models automatically
    - Gives actionable insights based on results
    """
    
    def __init__(self):
        """Initialize the evaluator with default settings."""
        self.results = {}
        self.model_type = None
    
    def evaluate_classification(self, model, X_test, y_test, model_name="Model", 
                              class_names=None, plot_curves=True):
        """
        Comprehensive evaluation for classification models.
        
        What this evaluates:
        - Accuracy: Overall correctness (good for balanced datasets)
        - Precision: Of positive predictions, how many were correct? (important when false positives are costly)
        - Recall: Of actual positives, how many were found? (important when false negatives are costly)
        - F1-Score: Balance between precision and recall
        - ROC-AUC: Overall ranking ability (good for imbalanced datasets)
        
        Parameters:
        - model: trained sklearn model
        - X_test: test features
        - y_test: test labels
        - model_name: name for this model
        - class_names: names for classes (optional)
        - plot_curves: whether to plot ROC and PR curves
        
        When to use each metric:
        - Medical diagnosis: High recall (don't miss sick patients)
        - Spam detection: High precision (don't mark important emails as spam)
        - Balanced problems: Accuracy or F1-score
        - Imbalanced problems: ROC-AUC or F1-score
        """
        print(f"\n🎯 Classification Evaluation: {model_name}")
        print("="*60)
        
        # Make predictions
        print("📊 Making predictions...")
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Get probabilities if available
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]  # For binary classification
            has_proba = True
        except:
            y_pred_proba = None
            has_proba = False
        
        # Calculate all metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'prediction_time': prediction_time,
            'samples_evaluated': len(y_test)
        }
        
        # Add ROC-AUC if probabilities available
        if has_proba and len(np.unique(y_test)) == 2:  # Binary classification
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        self.results[model_name] = metrics
        self.model_type = 'classification'
        
        # Print results with explanations
        print(f"\n📈 Performance Metrics:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f} - Overall correctness (% of correct predictions)")
        print(f"   Precision: {metrics['precision']:.4f} - Of positive predictions, % that were correct")
        print(f"   Recall:    {metrics['recall']:.4f} - Of actual positives, % that were found")
        print(f"   F1-Score:  {metrics['f1_score']:.4f} - Harmonic mean of precision and recall")
        
        if 'roc_auc' in metrics:
            print(f"   ROC-AUC:   {metrics['roc_auc']:.4f} - Model's ranking ability (0.5=random, 1.0=perfect)")
        
        print(f"\n⏱️ Performance:")
        print(f"   Prediction time: {prediction_time:.4f} seconds")
        print(f"   Speed: {len(y_test)/prediction_time:.0f} predictions/second")
        
        # Detailed classification report
        print(f"\n📊 Detailed Classification Report:")
        if class_names:
            print(classification_report(y_test, y_pred, target_names=class_names))
        else:
            print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        self._plot_confusion_matrix(y_test, y_pred, class_names, model_name)
        
        # Plot ROC and PR curves if possible
        if plot_curves and has_proba and len(np.unique(y_test)) == 2:
            self._plot_roc_curve(y_test, y_pred_proba, model_name)
            self._plot_precision_recall_curve(y_test, y_pred_proba, model_name)
        
        # Provide interpretation
        self._interpret_classification_results(metrics)
        
        return metrics
    
    def evaluate_regression(self, model, X_test, y_test, model_name="Model", plot_residuals=True):
        """
        Comprehensive evaluation for regression models.
        
        What this evaluates:
        - R² Score: Proportion of variance explained (higher is better, max=1.0)
        - Mean Squared Error (MSE): Average squared difference (lower is better)
        - Root Mean Squared Error (RMSE): MSE in original units (interpretable)
        - Mean Absolute Error (MAE): Average absolute difference (robust to outliers)
        
        Parameters:
        - model: trained sklearn model
        - X_test: test features
        - y_test: test target values
        - model_name: name for this model
        - plot_residuals: whether to plot residual analysis
        
        How to interpret:
        - R² = 0.8: Model explains 80% of variance (good)
        - R² = 0.0: Model is no better than predicting the mean
        - R² < 0: Model is worse than predicting the mean
        - RMSE in same units as target (e.g., if predicting price in $, RMSE is in $)
        - MAE less sensitive to outliers than RMSE
        """
        print(f"\n📈 Regression Evaluation: {model_name}")
        print("="*60)
        
        # Make predictions
        print("📊 Making predictions...")
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'prediction_time': prediction_time,
            'samples_evaluated': len(y_test)
        }
        
        # Store results
        self.results[model_name] = metrics
        self.model_type = 'regression'
        
        # Print results with explanations
        print(f"\n📈 Performance Metrics:")
        print(f"   R² Score:  {metrics['r2_score']:.4f} - Proportion of variance explained (closer to 1.0 = better)")
        print(f"   RMSE:      {metrics['rmse']:.4f} - Root Mean Squared Error (same units as target)")
        print(f"   MAE:       {metrics['mae']:.4f} - Mean Absolute Error (average prediction error)")
        print(f"   MSE:       {metrics['mse']:.4f} - Mean Squared Error (penalizes large errors more)")
        
        print(f"\n⏱️ Performance:")
        print(f"   Prediction time: {prediction_time:.4f} seconds")
        print(f"   Speed: {len(y_test)/prediction_time:.0f} predictions/second")
        
        # Additional statistics
        residuals = y_test - y_pred
        print(f"\n📊 Residual Analysis:")
        print(f"   Mean residual: {residuals.mean():.4f} (should be close to 0)")
        print(f"   Std of residuals: {residuals.std():.4f}")
        print(f"   Min residual: {residuals.min():.4f}")
        print(f"   Max residual: {residuals.max():.4f}")
        
        # Plot residuals if requested
        if plot_residuals:
            self._plot_residuals(y_test, y_pred, model_name)
        
        # Provide interpretation
        self._interpret_regression_results(metrics, y_test)
        
        return metrics
    
    def cross_validate_model(self, model, X, y, cv=5, scoring=None):
        """
        Perform cross-validation to get robust performance estimates.
        
        What this does:
        - Splits data into k folds (default 5)
        - Trains model on k-1 folds, tests on remaining fold
        - Repeats k times with different test fold each time
        - Reports mean and standard deviation of performance
        
        Why cross-validation:
        - Single train/test split might be lucky or unlucky
        - Cross-validation gives more reliable estimate
        - Helps detect high variance in model performance
        - Standard practice for model selection
        
        How to interpret results:
        - High mean score: Good average performance
        - Low standard deviation: Consistent performance
        - High standard deviation: Unstable model (might overfit)
        """
        print(f"\n🔄 Cross-Validation Analysis")
        print("="*50)
        
        # Determine scoring metric
        if scoring is None:
            if len(np.unique(y)) < 20:  # Classification heuristic
                scoring = 'accuracy'
            else:
                scoring = 'r2'
        
        print(f"📊 Performing {cv}-fold cross-validation with '{scoring}' metric...")
        
        # Perform cross-validation
        start_time = time.time()
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        cv_time = time.time() - start_time
        
        # Calculate statistics
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        min_score = cv_scores.min()
        max_score = cv_scores.max()
        
        print(f"\n📈 Cross-Validation Results:")
        print(f"   Mean {scoring}: {mean_score:.4f} ± {std_score:.4f}")
        print(f"   Min {scoring}:  {min_score:.4f}")
        print(f"   Max {scoring}:  {max_score:.4f}")
        print(f"   Range:         {max_score - min_score:.4f}")
        print(f"   CV Time:       {cv_time:.2f} seconds")
        
        # Individual fold results
        print(f"\n📋 Individual Fold Results:")
        for i, score in enumerate(cv_scores, 1):
            print(f"   Fold {i}: {score:.4f}")
        
        # Interpretation
        coefficient_of_variation = std_score / abs(mean_score) if mean_score != 0 else float('inf')
        
        print(f"\n🔍 Stability Analysis:")
        print(f"   Coefficient of Variation: {coefficient_of_variation:.4f}")
        
        if coefficient_of_variation < 0.1:
            print("   ✅ Very stable model - consistent performance across folds")
        elif coefficient_of_variation < 0.2:
            print("   ✅ Reasonably stable model")
        else:
            print("   ⚠️ Unstable model - high variance in performance")
            print("      Consider: regularization, more data, or simpler model")
        
        return {
            'mean_score': mean_score,
            'std_score': std_score,
            'cv_scores': cv_scores,
            'scoring_metric': scoring
        }
    
    def compare_models(self, models_dict, X_test, y_test, X_train=None, y_train=None):
        """
        Compare multiple models side by side.
        
        What this does:
        - Evaluates all models on the same test set
        - Compares performance metrics
        - Ranks models by performance
        - Provides recommendations
        
        Parameters:
        - models_dict: dictionary of {name: trained_model}
        - X_test, y_test: test data
        - X_train, y_train: optional training data for additional analysis
        """
        print(f"\n🏆 Model Comparison")
        print("="*60)
        
        comparison_results = {}
        
        # Evaluate each model
        for name, model in models_dict.items():
            print(f"\n📊 Evaluating {name}...")
            
            if self.model_type == 'classification' or len(np.unique(y_test)) < 20:
                metrics = self.evaluate_classification(model, X_test, y_test, name, plot_curves=False)
            else:
                metrics = self.evaluate_regression(model, X_test, y_test, name, plot_residuals=False)
            
            comparison_results[name] = metrics
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results).T
        
        # Sort by primary metric
        if self.model_type == 'classification':
            primary_metric = 'f1_score'
        else:
            primary_metric = 'r2_score'
        
        comparison_df = comparison_df.sort_values(primary_metric, ascending=False)
        
        print(f"\n🏆 Model Ranking (by {primary_metric}):")
        print("="*60)
        print(comparison_df.round(4))
        
        # Highlight best model
        best_model = comparison_df.index[0]
        best_score = comparison_df.loc[best_model, primary_metric]
        
        print(f"\n🥇 Best Model: {best_model}")
        print(f"   {primary_metric}: {best_score:.4f}")
        
        # Performance visualization
        self._plot_model_comparison(comparison_df, primary_metric)
        
        return comparison_df
    
    def hyperparameter_tuning(self, model, param_grid, X_train, y_train, 
                             search_type='grid', cv=5, n_iter=100):
        """
        Perform hyperparameter tuning with detailed explanations.
        
        What this does:
        - Tests different parameter combinations
        - Uses cross-validation to evaluate each combination
        - Finds the best parameter set
        - Returns optimized model
        
        Search Types:
        - Grid Search: Tests all combinations (thorough but slow)
        - Random Search: Tests random combinations (faster, often just as good)
        
        Parameters:
        - model: sklearn model to tune
        - param_grid: dictionary of parameters to try
        - X_train, y_train: training data
        - search_type: 'grid' or 'random'
        - cv: cross-validation folds
        - n_iter: number of iterations for random search
        """
        print(f"\n🔧 Hyperparameter Tuning ({search_type.title()} Search)")
        print("="*60)
        
        # Determine scoring metric
        if len(np.unique(y_train)) < 20:  # Classification
            scoring = 'f1_weighted'
        else:
            scoring = 'r2'
        
        print(f"📊 Search Configuration:")
        print(f"   Search type: {search_type}")
        print(f"   Cross-validation folds: {cv}")
        print(f"   Scoring metric: {scoring}")
        
        if search_type == 'grid':
            # Calculate total combinations
            total_combinations = 1
            for param, values in param_grid.items():
                total_combinations *= len(values)
                print(f"   {param}: {len(values)} values")
            print(f"   Total combinations: {total_combinations}")
            print(f"   Total models to train: {total_combinations * cv}")
            
            # Perform grid search
            search = GridSearchCV(
                model, param_grid, cv=cv, scoring=scoring,
                n_jobs=-1, verbose=1, return_train_score=True
            )
        else:
            print(f"   Random iterations: {n_iter}")
            print(f"   Total models to train: {n_iter * cv}")
            
            # Perform random search
            search = RandomizedSearchCV(
                model, param_grid, n_iter=n_iter, cv=cv, scoring=scoring,
                n_jobs=-1, verbose=1, return_train_score=True, random_state=42
            )
        
        # Fit the search
        print(f"\n🚀 Starting hyperparameter search...")
        start_time = time.time()
        search.fit(X_train, y_train)
        search_time = time.time() - start_time
        
        print(f"\n✅ Search completed in {search_time:.2f} seconds")
        
        # Results
        print(f"\n🏆 Best Parameters:")
        for param, value in search.best_params_.items():
            print(f"   {param}: {value}")
        
        print(f"\n📈 Performance:")
        print(f"   Best CV Score: {search.best_score_:.4f}")
        print(f"   Best CV Std:   {search.cv_results_['std_test_score'][search.best_index_]:.4f}")
        
        # Compare with default parameters
        if hasattr(model, 'get_params'):
            default_model = model.__class__()
            default_score = cross_val_score(default_model, X_train, y_train, 
                                          cv=cv, scoring=scoring).mean()
            improvement = search.best_score_ - default_score
            print(f"   Default Score: {default_score:.4f}")
            print(f"   Improvement:   {improvement:.4f} ({improvement/abs(default_score)*100:.1f}%)")
        
        # Plot parameter importance if possible
        self._plot_parameter_importance(search)
        
        return search.best_estimator_, search.best_params_, search.best_score_
    
    def _plot_confusion_matrix(self, y_true, y_pred, class_names, model_name):
        """Plot confusion matrix with detailed annotations."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
    
    def _plot_roc_curve(self, y_true, y_pred_proba, model_name):
        """Plot ROC curve for binary classification."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def _plot_precision_recall_curve(self, y_true, y_pred_proba, model_name):
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'{model_name}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def _plot_residuals(self, y_true, y_pred, model_name):
        """Plot residual analysis for regression."""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Residuals vs Predictions
        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title(f'Residuals vs Predictions - {model_name}')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1].axvline(residuals.mean(), color='r', linestyle='--', 
                       label=f'Mean: {residuals.mean():.3f}')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Residuals Distribution - {model_name}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_model_comparison(self, comparison_df, primary_metric):
        """Create visualization comparing model performance."""
        plt.figure(figsize=(12, 6))
        
        # Bar plot of primary metric
        bars = plt.bar(range(len(comparison_df)), comparison_df[primary_metric])
        plt.xticks(range(len(comparison_df)), comparison_df.index, rotation=45)
        plt.ylabel(primary_metric.replace('_', ' ').title())
        plt.title(f'Model Comparison - {primary_metric.replace("_", " ").title()}')
        
        # Add value labels on bars
        for i, (model, score) in enumerate(zip(comparison_df.index, comparison_df[primary_metric])):
            plt.text(i, score + max(comparison_df[primary_metric]) * 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Color the best model differently
        bars[0].set_color('gold')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_parameter_importance(self, search):
        """Plot parameter importance from hyperparameter search."""
        # This is a simplified version - in practice, you'd need more sophisticated analysis
        results_df = pd.DataFrame(search.cv_results_)
        
        # Extract parameter columns
        param_cols = [col for col in results_df.columns if col.startswith('param_')]
        
        if len(param_cols) <= 2:  # Simple visualization for 1-2 parameters
            plt.figure(figsize=(10, 6))
            
            if len(param_cols) == 1:
                # Single parameter
                param_name = param_cols[0].replace('param_', '')
                plt.scatter(results_df[param_cols[0]], results_df['mean_test_score'])
                plt.xlabel(param_name)
                plt.ylabel('CV Score')
                plt.title(f'Parameter Tuning Results - {param_name}')
            
            else:
                # Two parameters - create heatmap if possible
                param1, param2 = param_cols
                param1_name = param1.replace('param_', '')
                param2_name = param2.replace('param_', '')
                
                # Create pivot table
                pivot_data = results_df.pivot_table(
                    values='mean_test_score', 
                    index=param1, 
                    columns=param2, 
                    aggfunc='mean'
                )
                
                sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis')
                plt.title(f'Parameter Tuning Heatmap - {param1_name} vs {param2_name}')
            
            plt.tight_layout()
            plt.show()
    
    def _interpret_classification_results(self, metrics):
        """Provide interpretation of classification results."""
        print(f"\n💡 Interpretation & Recommendations:")
        
        accuracy = metrics['accuracy']
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1_score']
        
        if accuracy > 0.9:
            print("   ✅ Excellent accuracy! Model performs very well.")
        elif accuracy > 0.8:
            print("   ✅ Good accuracy. Model is performing well.")
        elif accuracy > 0.7:
            print("   ⚠️ Moderate accuracy. Consider improving features or trying different algorithms.")
        else:
            print("   🚨 Low accuracy. Model needs significant improvement.")
        
        # Precision vs Recall analysis
        if precision > recall + 0.1:
            print("   📊 High precision, lower recall - model is conservative")
            print("      Good when false positives are costly")
        elif recall > precision + 0.1:
            print("   📊 High recall, lower precision - model is liberal") 
            print("      Good when false negatives are costly")
        else:
            print("   📊 Balanced precision and recall - well-rounded model")
        
        # F1-Score interpretation
        if f1 > 0.8:
            print("   🎯 High F1-score indicates good balance of precision and recall")
        elif f1 < 0.6:
            print("   🎯 Low F1-score suggests model struggles with both precision and recall")
    
    def _interpret_regression_results(self, metrics, y_test):
        """Provide interpretation of regression results."""
        print(f"\n💡 Interpretation & Recommendations:")
        
        r2 = metrics['r2_score']
        rmse = metrics['rmse']
        mae = metrics['mae']
        
        # R² interpretation
        if r2 > 0.9:
            print("   ✅ Excellent R² score! Model explains >90% of variance.")
        elif r2 > 0.7:
            print("   ✅ Good R² score. Model explains most of the variance.")
        elif r2 > 0.5:
            print("   ⚠️ Moderate R² score. Model has some predictive power.")
        elif r2 > 0:
            print("   🚨 Low R² score. Model barely better than predicting the mean.")
        else:
            print("   🚨 Negative R² score. Model is worse than predicting the mean!")
        
        # Error interpretation
        target_range = y_test.max() - y_test.min()
        rmse_percentage = (rmse / target_range) * 100
        mae_percentage = (mae / target_range) * 100
        
        print(f"   📊 RMSE is {rmse_percentage:.1f}% of target range")
        print(f"   📊 MAE is {mae_percentage:.1f}% of target range")
        
        if rmse_percentage < 10:
            print("   ✅ Very low error relative to target range")
        elif rmse_percentage < 20:
            print("   ✅ Reasonable error for most applications")
        else:
            print("   ⚠️ High error - consider feature engineering or different algorithms")

# Utility functions
def quick_model_evaluation(model, X_test, y_test, model_name="Model"):
    """Quick evaluation function for any model."""
    evaluator = ModelEvaluator()
    
    # Determine problem type
    if len(np.unique(y_test)) < 20:  # Classification
        return evaluator.evaluate_classification(model, X_test, y_test, model_name)
    else:  # Regression
        return evaluator.evaluate_regression(model, X_test, y_test, model_name)

if __name__ == "__main__":
    print("ModelEvaluator Demonstration")
    print("="*50)
    
    # This would contain example usage with sample data
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_classification(model, X_test, y_test, "Random Forest")