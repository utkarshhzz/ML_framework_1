# 📏 Performance Metrics Guide
*Complete Guide to ML Evaluation Metrics*

## 🎯 Overview

This guide provides comprehensive explanations of all performance metrics used in our ML framework. Understanding these metrics is crucial for properly evaluating and comparing machine learning models.

---

## 🎯 Classification Metrics

### 📊 Confusion Matrix - The Foundation

The confusion matrix is the foundation of all classification metrics. It's a table showing correct vs predicted classifications.

```
                    PREDICTED
                 Positive  Negative
ACTUAL Positive    TP       FN
       Negative    FP       TN
```

**What each cell means:**
- **TP (True Positive)**: Correctly predicted positive cases
- **TN (True Negative)**: Correctly predicted negative cases  
- **FP (False Positive)**: Incorrectly predicted as positive (Type I Error)
- **FN (False Negative)**: Incorrectly predicted as negative (Type II Error)

**Example - Medical Test:**
```
Disease Test Results (1000 patients)
                    PREDICTED
                 Sick  Healthy
ACTUAL    Sick    85      15     (100 actually sick)
       Healthy    20     880     (900 actually healthy)
```

**Interpretation:**
- 85 sick patients correctly identified
- 15 sick patients missed (dangerous!)
- 20 healthy patients incorrectly flagged (unnecessary worry)
- 880 healthy patients correctly identified

---

### 🎯 Accuracy

**Formula:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**What it measures:** Overall correctness - percentage of all predictions that were correct.

**Example calculation:**
```
Accuracy = (85 + 880) / (85 + 880 + 20 + 15) = 965/1000 = 0.965 = 96.5%
```

**When to use:**
- ✅ Balanced datasets (roughly equal class sizes)
- ✅ All errors are equally costly
- ✅ Quick overall performance check

**When NOT to use:**
- ❌ Imbalanced datasets (e.g., 95% one class, 5% other)
- ❌ False positives and false negatives have different costs
- ❌ Rare event detection (fraud, disease diagnosis)

**Why accuracy fails with imbalanced data:**
```
Fraud Detection Dataset: 9900 legitimate, 100 fraudulent transactions

Naive model that always predicts "legitimate":
Accuracy = 9900/10000 = 99%

But this model catches 0% of fraud! Accuracy is misleading.
```

---

### 🎯 Precision

**Formula:**
```
Precision = TP / (TP + FP)
```

**What it measures:** Of all positive predictions, how many were actually correct?

**Intuitive explanation:** "When the model says YES, how often is it right?"

**Example calculation:**
```
Precision = 85 / (85 + 20) = 85/105 = 0.809 = 80.9%
```

**Real-world interpretation:**
- Email spam filter: Of emails marked as spam, 80.9% actually were spam
- Medical test: Of patients diagnosed as sick, 80.9% actually were sick

**When precision is critical:**
- **False positives are costly or harmful**
- Email spam (don't want important emails in spam folder)
- Criminal justice (don't want to wrongly convict innocent people)
- Drug approval (don't want to approve harmful drugs)
- Marketing campaigns (don't want to waste money on wrong targets)

**Trade-off with recall:**
- High precision often means low recall
- Being very careful about positive predictions means missing some actual positives

---

### 🎯 Recall (Sensitivity)

**Formula:**
```
Recall = TP / (TP + FN)
```

**What it measures:** Of all actual positive cases, how many did we correctly identify?

**Intuitive explanation:** "When the answer should be YES, how often do we catch it?"

**Example calculation:**
```
Recall = 85 / (85 + 15) = 85/100 = 0.85 = 85%
```

**Real-world interpretation:**
- Medical test: Of all actually sick patients, we correctly identified 85%
- Fraud detection: Of all actual fraud cases, we caught 85%

**When recall is critical:**
- **False negatives are costly or dangerous**
- Medical diagnosis (don't want to miss sick patients)
- Security screening (don't want to miss threats)
- Fraud detection (don't want to miss fraudulent transactions)
- Search engines (don't want to miss relevant results)

**Trade-off with precision:**
- High recall often means low precision
- Casting a wide net to catch all positives means catching some false positives too

---

### 🎯 F1-Score

**Formula:**
```
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

**What it measures:** Harmonic mean of precision and recall - balances both metrics.

**Why harmonic mean:** Gives more weight to smaller values, so both precision and recall need to be reasonably high for a good F1-score.

**Example calculation:**
```
F1 = 2 × (0.809 × 0.85) / (0.809 + 0.85) = 2 × 0.688 / 1.659 = 0.829 = 82.9%
```

**When to use F1-Score:**
- ✅ Need balance between precision and recall
- ✅ Imbalanced datasets
- ✅ Both false positives and false negatives are problematic
- ✅ Want single metric that considers both precision and recall

**F1-Score interpretations:**
- F1 = 1.0: Perfect precision AND recall
- F1 = 0.0: Either precision or recall (or both) is zero
- F1 > 0.8: Generally considered good performance
- F1 < 0.5: Poor performance, model needs improvement

**Variants:**
- **F0.5-Score**: Weights precision higher than recall
- **F2-Score**: Weights recall higher than precision

---

### 🎯 Specificity

**Formula:**
```
Specificity = TN / (TN + FP)
```

**What it measures:** Of all actual negative cases, how many did we correctly identify?

**Intuitive explanation:** "When the answer should be NO, how often do we get it right?"

**Example calculation:**
```
Specificity = 880 / (880 + 20) = 880/900 = 0.978 = 97.8%
```

**When specificity matters:**
- Medical testing (correctly identifying healthy patients)
- Quality control (correctly identifying good products)
- Security systems (correctly identifying non-threats)

---

### 🎯 ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

**What ROC curve shows:**
- X-axis: False Positive Rate (1 - Specificity)
- Y-axis: True Positive Rate (Recall)
- Plots model performance at different classification thresholds

**AUC interpretation:**
- AUC = 1.0: Perfect classifier
- AUC = 0.9: Excellent performance
- AUC = 0.8: Good performance  
- AUC = 0.7: Fair performance
- AUC = 0.5: Random classifier (no better than coin flip)
- AUC < 0.5: Worse than random (but can invert predictions)

**Intuitive meaning of AUC:**
"If I randomly pick one positive example and one negative example, what's the probability that the model assigns a higher score to the positive example?"

**When to use ROC-AUC:**
- ✅ Binary classification
- ✅ Care about ranking/ordering of predictions
- ✅ Want threshold-independent metric
- ✅ Balanced or slightly imbalanced datasets

**When NOT to use ROC-AUC:**
- ❌ Highly imbalanced datasets (can be overly optimistic)
- ❌ Care more about precision than recall
- ❌ Multi-class problems (without modification)

---

### 🎯 Precision-Recall AUC

**What PR curve shows:**
- X-axis: Recall
- Y-axis: Precision
- Better alternative to ROC for imbalanced datasets

**When to use PR-AUC instead of ROC-AUC:**
- Highly imbalanced datasets
- Positive class is rare and important
- Care more about precision than specificity

**Example:**
```
Fraud detection with 99% legitimate transactions:
- ROC-AUC might be 0.95 (looks great!)
- PR-AUC might be 0.60 (more realistic assessment)
```

---

## 📈 Regression Metrics

### 🎯 Mean Absolute Error (MAE)

**Formula:**
```
MAE = (1/n) × Σ|yᵢ - ŷᵢ|
```

**What it measures:** Average absolute difference between actual and predicted values.

**Units:** Same as target variable (interpretable!)

**Example:**
```
House prices (in $1000s): Actual = [200, 300, 250, 400]
                         Predicted = [180, 320, 240, 450]
MAE = (|200-180| + |300-320| + |250-240| + |400-450|) / 4
    = (20 + 20 + 10 + 50) / 4 = 25

Interpretation: On average, predictions are off by $25,000
```

**Characteristics:**
- **Robust to outliers**: Large errors don't dominate
- **Linear**: All errors weighted equally
- **Interpretable**: Easy to understand and explain
- **Not differentiable**: Can cause issues with gradient-based optimization

**When to use MAE:**
- ✅ Want robust metric (not sensitive to outliers)
- ✅ All errors equally important
- ✅ Need interpretable results
- ✅ Target variable has meaningful units

---

### 🎯 Mean Squared Error (MSE)

**Formula:**
```
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
```

**What it measures:** Average squared difference between actual and predicted values.

**Units:** Square of target variable units

**Example:**
```
Same house prices: Actual = [200, 300, 250, 400]
                  Predicted = [180, 320, 240, 450]
MSE = ((200-180)² + (300-320)² + (250-240)² + (400-450)²) / 4
    = (400 + 400 + 100 + 2500) / 4 = 850

Units: ($1000)² - not directly interpretable
```

**Characteristics:**
- **Sensitive to outliers**: Large errors get amplified
- **Differentiable**: Good for optimization
- **Penalizes large errors more**: Quadratic penalty
- **Not in original units**: Harder to interpret

**When to use MSE:**
- ✅ Large errors are particularly bad
- ✅ Using gradient-based optimization
- ✅ Mathematical convenience needed
- ✅ Normally distributed errors expected

---

### 🎯 Root Mean Squared Error (RMSE)

**Formula:**
```
RMSE = √MSE = √[(1/n) × Σ(yᵢ - ŷᵢ)²]
```

**What it measures:** Square root of MSE, bringing units back to original scale.

**Units:** Same as target variable (interpretable again!)

**Example:**
```
RMSE = √850 = 29.15

Interpretation: On average, predictions are off by $29,150
```

**Characteristics:**
- **More sensitive to outliers than MAE**: But less than MSE
- **Same units as target**: Interpretable
- **Standard metric**: Widely used and understood
- **Penalizes large errors**: More than MAE, less than MSE

**RMSE vs MAE comparison:**
```
If RMSE >> MAE: Large errors present (outliers)
If RMSE ≈ MAE: Errors are uniform in size
```

---

### 🎯 R² Score (Coefficient of Determination)

**Formula:**
```
R² = 1 - (SS_res / SS_tot)

Where:
SS_res = Σ(yᵢ - ŷᵢ)²     (Sum of Squared Residuals)
SS_tot = Σ(yᵢ - ȳ)²      (Total Sum of Squares)
```

**What it measures:** Proportion of variance in target variable explained by the model.

**Intuitive explanation:** "How much better is my model than just predicting the mean?"

**Range:** Can be any value ≤ 1
- R² = 1.0: Perfect predictions
- R² = 0.8: Model explains 80% of variance
- R² = 0.0: Model is as good as predicting mean
- R² < 0.0: Model is worse than predicting mean

**Example:**
```
House prices: Mean = $275k
Baseline error (always predict mean): High variance
Model error: Lower variance
R² = 0.85 means model explains 85% of price variation
```

**When to use R²:**
- ✅ Want percentage of variance explained
- ✅ Compare models on same dataset
- ✅ Understand model's explanatory power
- ✅ Unitless metric needed

**When NOT to use R²:**
- ❌ Compare models on different datasets
- ❌ Care about absolute error size
- ❌ Non-linear relationships with different variance patterns

---

## 🎯 Multi-Class Classification Metrics

### 📊 Macro vs Micro vs Weighted Averaging

When dealing with multiple classes, we need to decide how to combine per-class metrics.

**Macro Average:**
```
Macro-F1 = (F1_class1 + F1_class2 + F1_class3) / 3
```
- Treats all classes equally
- Good when all classes are equally important
- Can be dominated by performance on rare classes

**Micro Average:**
```
Micro-F1 = F1_score calculated from global TP, FP, FN
```
- Weighted by class frequency
- Good when larger classes are more important
- Similar to accuracy for balanced datasets

**Weighted Average:**
```
Weighted-F1 = (F1_class1 × n1 + F1_class2 × n2 + F1_class3 × n3) / (n1+n2+n3)
```
- Weighted by class support (number of samples)
- Good compromise between macro and micro

**Example:**
```
Classes: A (1000 samples), B (100 samples), C (10 samples)
F1 scores: A=0.9, B=0.8, C=0.5

Macro-F1 = (0.9 + 0.8 + 0.5) / 3 = 0.73
Micro-F1 ≈ 0.88 (dominated by class A)
Weighted-F1 ≈ 0.87 (close to micro, but considers all classes)
```

---

## 🔍 Choosing the Right Metric

### 🎯 Decision Framework

**Step 1: Problem Type**
- Classification → Accuracy, Precision, Recall, F1, ROC-AUC
- Regression → MAE, RMSE, R²

**Step 2: Class Balance (Classification)**
- Balanced → Accuracy, ROC-AUC
- Imbalanced → F1-Score, PR-AUC, Precision/Recall

**Step 3: Error Costs**
- Equal costs → Accuracy, F1-Score
- FP more costly → Precision
- FN more costly → Recall
- Large errors very bad → RMSE
- All errors equal → MAE

**Step 4: Interpretability**
- Need interpretation → MAE, Accuracy
- Math convenience → MSE, ROC-AUC

### 📊 Metric Selection Guide

| **Scenario** | **Best Metrics** | **Why** |
|--------------|------------------|---------|
| **Balanced classification** | Accuracy, F1-Score | Simple and interpretable |
| **Imbalanced classification** | F1-Score, PR-AUC | Handles class imbalance |
| **Medical diagnosis** | Recall, F1-Score | Don't miss positive cases |
| **Spam detection** | Precision, F1-Score | Don't flag important emails |
| **Ranking problems** | ROC-AUC | Cares about ordering |
| **House price prediction** | MAE, RMSE | Interpretable error in $ |
| **Outlier-prone regression** | MAE | Robust to extreme values |
| **Normal error regression** | RMSE, R² | Standard and interpretable |

---

## 📈 Practical Implementation

### 🎯 Scikit-learn Implementation

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
)

# Classification metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')  
f1 = f1_score(y_true, y_pred, average='weighted')
auc = roc_auc_score(y_true, y_pred_proba)

# Regression metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)
```

### 🎯 Custom Metric Function

```python
def comprehensive_classification_report(y_true, y_pred, y_pred_proba=None):
    """
    Generate comprehensive classification metrics with explanations.
    """
    print("📊 Classification Performance Report")
    print("="*50)
    
    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"Accuracy:  {acc:.4f} - Overall correctness")
    print(f"Precision: {prec:.4f} - Of positive predictions, % correct")
    print(f"Recall:    {rec:.4f} - Of actual positives, % found")
    print(f"F1-Score:  {f1:.4f} - Balance of precision and recall")
    
    # ROC-AUC if probabilities available
    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_pred_proba)
        print(f"ROC-AUC:   {auc:.4f} - Ranking ability (0.5=random, 1.0=perfect)")
    
    return {
        'accuracy': acc, 'precision': prec, 
        'recall': rec, 'f1_score': f1
    }
```

---

## 🎯 Common Pitfalls and Solutions

### ❌ Common Mistakes

1. **Using accuracy on imbalanced data**
   - Problem: 99% accuracy might mean 0% recall on minority class
   - Solution: Use F1-score or PR-AUC

2. **Ignoring class imbalance in metrics**
   - Problem: Macro-average dominated by rare classes
   - Solution: Use weighted or micro averaging

3. **Not considering business costs**
   - Problem: Optimizing wrong metric for business impact
   - Solution: Define cost matrix, use appropriate metric

4. **Comparing R² across different datasets**
   - Problem: R² depends on data variance
   - Solution: Use RMSE or MAE for cross-dataset comparison

5. **Using MSE when outliers present**
   - Problem: Few large errors dominate metric
   - Solution: Use MAE or robust metrics

### ✅ Best Practices

1. **Always report multiple metrics**
2. **Understand your business problem first**
3. **Consider class balance in your data**
4. **Use cross-validation for reliable estimates**
5. **Plot confusion matrices for classification**
6. **Plot residuals for regression**
7. **Report confidence intervals when possible**

---

## 🎯 Conclusion

**Key Takeaways:**

1. **No single metric is perfect** - Always use multiple metrics
2. **Context matters** - Choose metrics based on your specific problem
3. **Understand trade-offs** - Precision vs Recall, Accuracy vs Interpretability
4. **Business impact** - Align metrics with business objectives
5. **Validate properly** - Use cross-validation for robust evaluation

**Metric Selection Hierarchy:**
1. **Business requirements** - What matters for your application?
2. **Data characteristics** - Balanced vs imbalanced, outliers, etc.
3. **Model properties** - What does your model output?
4. **Interpretability needs** - Do stakeholders need to understand?

Remember: The best model is not always the one with the highest metric score, but the one that best solves your specific problem!

---

*This guide provides the foundation for proper model evaluation. Use it alongside the practical implementations in our framework notebooks.*