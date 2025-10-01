# 📚 Machine Learning Mathematics Guide
*Complete Mathematical Foundations for ML Framework*

## 🎯 Overview

This guide provides comprehensive mathematical explanations for all algorithms and concepts used in our ML framework. Each section includes the mathematical formulation, intuitive explanation, and practical implementation details.

---

## 📊 1. AdaBoost (Adaptive Boosting)

### 🧮 Mathematical Foundation

#### AdaBoost Algorithm
AdaBoost combines multiple weak learners sequentially, where each learner focuses on previously misclassified examples.

**Core Algorithm Steps:**

1. **Initialize sample weights:**
```
w₁(i) = 1/N  for i = 1, ..., N
```

2. **For each boosting round t = 1, ..., T:**

   a) **Train weak learner hₜ with weighted samples**
   
   b) **Calculate weighted error:**
   ```
   εₜ = ∑ᵢ w₁(i) · I(hₜ(xᵢ) ≠ yᵢ)
   ```
   
   c) **Calculate classifier weight:**
   ```
   αₛ = (1/2) · ln((1 - εₜ)/εₜ)
   ```
   
   d) **Update sample weights:**
   ```
   wₜ₊₁(i) = wₜ(i) · exp(-αₛ · yᵢ · hₜ(xᵢ)) / Zₜ
   ```
   where Zₜ is normalization factor

3. **Final classifier:**
```
H(x) = sign(∑ₜ₌₁ᵀ αₜ · hₜ(x))
```

#### Key Mathematical Insights:

**Weight Update Mechanism:**
- **Correctly classified**: wₜ₊₁(i) = wₜ(i) · exp(-αₜ) (weight decreases)
- **Misclassified**: wₜ₊₁(i) = wₜ(i) · exp(αₜ) (weight increases)
- **Effect**: Forces next classifier to focus on difficult examples

**Classifier Weight (αₜ):**
- **Low error (εₜ → 0)**: αₜ → ∞ (high influence)
- **High error (εₜ → 0.5)**: αₜ → 0 (low influence)
- **Random guessing (εₜ = 0.5)**: αₜ = 0 (no contribution)

### 🎯 Key Parameters Explained

#### n_estimators (Number of Weak Learners)
```
Mathematical Impact: H(x) = sign(∑ₜ₌₁ⁿ_ᵉˢᵗⁱᵐᵃᵗᵒʳˢ αₜ · hₜ(x))
```
- **Small values (10-50)**: Fast training, potential underfitting
- **Large values (100-500)**: Better performance, overfitting risk
- **Trade-off**: Bias vs. Variance, Training time vs. Accuracy

#### learning_rate (Shrinkage Parameter)
```
Modified classifier weight: α'ₜ = learning_rate × αₜ
Final prediction: H(x) = sign(∑ₜ₌₁ᵀ α'ₜ · hₜ(x))
```
- **High rates (1.0-2.0)**: Aggressive learning, fast convergence
- **Low rates (0.1-0.5)**: Conservative learning, better generalization
- **Mathematical effect**: Scales the contribution of each weak learner

#### algorithm Parameter
**SAMME (Stagewise Additive Modeling using Multi-class Exponential loss):**
```
Uses discrete class predictions: hₜ(x) ∈ {-1, +1}
Weight update: standard AdaBoost formula
```

**SAMME.R (Real SAMME):**
```
Uses class probabilities: pₜ(x) ∈ [0, 1]
Faster convergence through probability estimates
Better performance in most cases
```

### 🔍 Loss Function Analysis

#### Exponential Loss (Classification)
```
L(y, f(x)) = exp(-y · f(x))
```
- **Correct prediction (y·f(x) > 0)**: Loss < 1
- **Incorrect prediction (y·f(x) < 0)**: Loss > 1
- **Property**: Exponentially penalizes misclassifications

#### Regression Loss Functions
**Linear Loss:**
```
L(y, f(x)) = |y - f(x)|
```

**Square Loss:**
```
L(y, f(x)) = (y - f(x))²
```

**Exponential Loss:**
```
L(y, f(x)) = exp(|y - f(x)|)
```

### 🎓 Theoretical Properties

#### Generalization Bound
```
P(error) ≤ ∏ₜ₌₁ᵀ 2√(εₜ(1 - εₜ))
```
- Shows exponential decrease in training error
- Explains AdaBoost's resistance to overfitting
- Connects weak learning to strong learning

#### Margin Theory
```
Margin(x) = (y · ∑ₜ αₜhₜ(x)) / ∑ₜ αₜ
```
- AdaBoost tends to maximize margins
- Larger margins → better generalization
- Explains continued improvement after zero training error

## 📊 2. Support Vector Machines (SVM)

### 🧮 Mathematical Foundation

#### Linear SVM Optimization Problem
```
Minimize: (1/2)||w||² + C∑ξᵢ

Subject to:
- yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ  for all i
- ξᵢ ≥ 0  for all i
```

**Where:**
- `w`: weight vector (defines the separating hyperplane)
- `b`: bias term (shifts the hyperplane)
- `C`: regularization parameter (controls trade-off)
- `ξᵢ`: slack variables (allow for misclassification)
- `xᵢ, yᵢ`: training samples and labels

#### What Each Component Does:

**Objective Function: (1/2)||w||²**
- **Purpose**: Maximize margin between classes
- **Math**: ||w||² = w₁² + w₂² + ... + wₙ²
- **Intuition**: Smaller weights → larger margin → better generalization
- **Why 1/2**: Mathematical convenience for derivatives

**Regularization Term: C∑ξᵢ**
- **Purpose**: Control tolerance for misclassification
- **High C**: Low tolerance (hard margin, might overfit)
- **Low C**: High tolerance (soft margin, might underfit)
- **ξᵢ = 0**: Point is correctly classified with sufficient margin
- **0 < ξᵢ < 1**: Point is correctly classified but within margin
- **ξᵢ ≥ 1**: Point is misclassified

### 🎯 Kernel Functions

#### RBF (Radial Basis Function) Kernel
```
K(x, x') = exp(-γ||x - x'||²)
```

**Mathematical Breakdown:**
- `||x - x'||²`: Squared Euclidean distance between points
- `γ`: Kernel parameter (gamma)
  - High γ: Tight, complex decision boundaries
  - Low γ: Smooth, simple decision boundaries
- `exp()`: Exponential function creates smooth transitions

**How RBF Works:**
1. Calculate distance between every pair of points
2. Apply exponential decay: closer points → higher similarity
3. Create infinite-dimensional feature space implicitly
4. Enable non-linear decision boundaries in original space

#### Polynomial Kernel
```
K(x, x') = (γ⟨x, x'⟩ + r)ᵈ
```

**Parameters:**
- `d`: degree of polynomial (typically 2-4)
- `γ`: scaling parameter
- `r`: independent term
- `⟨x, x'⟩`: dot product of feature vectors

**Example (degree=2):**
- Original features: (x₁, x₂)
- Expanded features: (x₁², x₂², √2x₁x₂, √2x₁, √2x₂, 1)
- Creates quadratic decision boundaries

### 🔢 SVM Decision Function
```
f(x) = sign(∑αᵢyᵢK(xᵢ, x) + b)
```

**Components:**
- `αᵢ`: Lagrange multipliers (learned during training)
- `yᵢ`: class labels of support vectors
- `K(xᵢ, x)`: kernel function
- `b`: bias term
- `sign()`: outputs +1 or -1 for classification

**Support Vectors:**
- Training points with αᵢ > 0
- Points that define the decision boundary
- Typically only 5-20% of training data
- Removing non-support vectors doesn't change the model

---

## � 2. Naive Bayes

### 🧮 Mathematical Foundation

#### Bayes' Theorem
```
P(y|x) = P(x|y) × P(y) / P(x)
```

**For Classification:**
```
P(class|features) = P(features|class) × P(class) / P(features)
```

**Naive Independence Assumption:**
```
P(x₁, x₂, ..., xₙ|y) = P(x₁|y) × P(x₂|y) × ... × P(xₙ|y)
```

### 🎯 Gaussian Naive Bayes

#### Probability Density Function
```
P(xᵢ|y) = (1/√(2πσᵧ²)) × exp(-(xᵢ - μᵧ)²/(2σᵧ²))
```

**Parameters (learned from training data):**
- `μᵧ`: mean of feature xᵢ for class y
- `σᵧ²`: variance of feature xᵢ for class y

**Step-by-Step Calculation:**

1. **Learn Parameters (Training):**
   ```
   μᵧ = (1/nᵧ) × ∑(xᵢ where yᵢ = y)
   σᵧ² = (1/nᵧ) × ∑(xᵢ - μᵧ)² where yᵢ = y
   ```

2. **Calculate Class Probabilities (Prediction):**
   ```
   P(y) = count(y) / total_samples
   ```

3. **Calculate Feature Likelihoods:**
   ```
   For each feature xᵢ:
   P(xᵢ|y) = gaussian_pdf(xᵢ, μᵧ, σᵧ)
   ```

4. **Combine Using Naive Assumption:**
   ```
   P(y|x) ∝ P(y) × ∏P(xᵢ|y)
   ```

**Why "Naive":**
- Assumes features are independent given the class
- Often violated in practice, but works surprisingly well
- Example: In text classification, word occurrences are not independent
- Despite violation, often achieves good performance

### 🔢 Prediction Formula
```
ŷ = argmax P(y) × ∏P(xᵢ|y)
     y
```

**Practical Implementation (log probabilities):**
```
log P(y|x) = log P(y) + ∑log P(xᵢ|y)
```

**Why use log probabilities:**
- Avoids numerical underflow (multiplying many small numbers)
- Addition is faster than multiplication
- Monotonic transformation preserves ordering

---

## 🌳 3. Ensemble Methods

### 🎯 Random Forest Mathematics

#### Bagging (Bootstrap Aggregating)

**Bootstrap Sampling:**
- Create B bootstrap samples from original dataset
- Each sample: randomly select n samples with replacement
- Some samples appear multiple times, others not at all
- Reduces variance by averaging multiple models

**Random Feature Selection:**
- At each node split, randomly select m features from total p features
- Typical values: m = √p for classification, m = p/3 for regression
- Reduces correlation between trees
- Improves generalization

#### Prediction Aggregation

**Classification (Majority Voting):**
```
ŷ = mode{h₁(x), h₂(x), ..., hB(x)}
```

**Regression (Average):**
```
ŷ = (1/B) × ∑hᵦ(x)
        b=1
```

**Out-of-Bag (OOB) Error:**
```
OOB Error = (1/n) × ∑I(yᵢ ≠ ŷᵢ^OOB)
```
- Uses samples not included in bootstrap for validation
- No need for separate validation set
- Unbiased estimate of generalization error

### 🎯 AdaBoost Mathematics

#### Adaptive Boosting Algorithm

**Weight Update Rule:**
```
wᵢ^(t+1) = wᵢ^(t) × exp(-αₜyᵢhₜ(xᵢ)) / Zₜ
```

**Where:**
- `wᵢ^(t)`: weight of sample i at iteration t
- `αₜ`: weight of classifier t
- `hₜ(xᵢ)`: prediction of classifier t on sample i
- `Zₜ`: normalization constant

**Classifier Weight:**
```
αₜ = (1/2) × ln((1 - εₜ)/εₜ)
```

**Where:**
- `εₜ`: weighted error rate of classifier t
- Higher accuracy → higher weight in final prediction

**Final Prediction:**
```
H(x) = sign(∑αₜhₜ(x))
        t=1
```

**Key Insights:**
- Misclassified samples get higher weights
- Next classifier focuses on difficult samples
- Sequentially reduces training error
- Can overfit if not regularized

---

## 📏 4. Performance Metrics

### 🎯 Classification Metrics

#### Confusion Matrix Components
```
                Prediction
Actual    |  Pos  |  Neg  |
----------|-------|-------|
Positive  |  TP   |  FN   |
Negative  |  FP   |  TN   |
```

#### Metric Formulas

**Accuracy:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- Overall correctness
- Good for balanced datasets
- Misleading for imbalanced datasets

**Precision:**
```
Precision = TP / (TP + FP)
```
- Of positive predictions, how many were correct?
- Important when false positives are costly
- Example: Spam detection (don't want to mark important emails as spam)

**Recall (Sensitivity):**
```
Recall = TP / (TP + FN)
```
- Of actual positives, how many were found?
- Important when false negatives are costly
- Example: Medical diagnosis (don't want to miss sick patients)

**F1-Score:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- Harmonic mean of precision and recall
- Balances both metrics
- Good for imbalanced datasets

**Specificity:**
```
Specificity = TN / (TN + FP)
```
- True negative rate
- Important in medical testing

### 🎯 Regression Metrics

**Mean Squared Error (MSE):**
```
MSE = (1/n) × ∑(yᵢ - ŷᵢ)²
```
- Penalizes large errors more than small ones
- Same units as y²
- Always positive, 0 is perfect

**Root Mean Squared Error (RMSE):**
```
RMSE = √MSE = √[(1/n) × ∑(yᵢ - ŷᵢ)²]
```
- Same units as target variable
- Interpretable: average prediction error
- Sensitive to outliers

**Mean Absolute Error (MAE):**
```
MAE = (1/n) × ∑|yᵢ - ŷᵢ|
```
- Average absolute error
- Less sensitive to outliers than RMSE
- Same units as target variable

**R² Score (Coefficient of Determination):**
```
R² = 1 - SS_res/SS_tot

Where:
SS_res = ∑(yᵢ - ŷᵢ)²  (residual sum of squares)
SS_tot = ∑(yᵢ - ȳ)²   (total sum of squares)
```

**R² Interpretation:**
- R² = 1: Perfect predictions
- R² = 0: Model as good as predicting the mean
- R² < 0: Model worse than predicting the mean
- R² = 0.8: Model explains 80% of variance

---

## 🔧 5. Cross-Validation Mathematics

### 🎯 K-Fold Cross-Validation

**Process:**
1. Split data into k equal folds
2. For each fold i:
   - Train on k-1 folds
   - Test on fold i
   - Record performance score Sᵢ

**Final Score:**
```
CV_Score = (1/k) × ∑Sᵢ
CV_Std = √[(1/k) × ∑(Sᵢ - CV_Score)²]
```

**Bias-Variance Trade-off:**
- **Small k (e.g., k=3)**: 
  - Higher bias (less data for training)
  - Lower variance (more similar training sets)
  - Faster computation
  
- **Large k (e.g., k=10)**:
  - Lower bias (more data for training)
  - Higher variance (more different training sets)
  - Slower computation

**Leave-One-Out CV (LOOCV):**
- Special case: k = n (number of samples)
- Maximum training data for each fold
- Highest computational cost
- Can have high variance

### 🎯 Stratified Cross-Validation

**For Classification:**
- Maintains class distribution in each fold
- Ensures each fold is representative
- Reduces variance in performance estimates

**Mathematical Constraint:**
```
For each fold f and class c:
P(class=c in fold f) ≈ P(class=c in full dataset)
```

---

## 🎚️ 6. Hyperparameter Tuning Mathematics

### 🎯 Grid Search

**Search Space:**
```
If parameters p₁, p₂, ..., pₖ have n₁, n₂, ..., nₖ values respectively:
Total combinations = n₁ × n₂ × ... × nₖ
```

**With Cross-Validation:**
```
Total model fits = (∏nᵢ) × k_folds
```

**Computational Complexity:**
- Exponential in number of parameters
- Exhaustive but guaranteed to find optimum in search space
- Parallelizable across parameter combinations

### 🎯 Random Search

**Probability Theory:**
- For each iteration, randomly sample from parameter distributions
- More efficient than grid search for high dimensions
- Often finds good solutions faster

**Mathematical Justification:**
If optimal parameters lie in top 5% of search space:
- Grid search with 100 points: might miss optimal region
- Random search with 100 points: 99.4% chance to find good solution

**Advantages:**
- Better for continuous parameters
- Scales better with dimensionality
- Can run for any budget

---

*This guide serves as a reference for the mathematical concepts used throughout the ML framework. Each notebook provides practical implementations of these theoretical foundations.*
1. **Classification Problems** - Predicting categories
2. **Decision Trees** - Rule-based learning
3. **Data Visualization** - Understanding patterns
4. **Hyperparameter Tuning** - Optimizing models
5. **Confusion Matrix** - Classification evaluation

---

# 📖 NOTEBOOK 3: CALIFORNIA HOUSING REGRESSION

## Core Concepts You'll Master:
1. **Large Dataset Handling** - Sampling techniques
2. **Advanced Regression** - Decision tree regressors
3. **Grid Search** - Systematic optimization
4. **Model Comparison** - Before/after tuning
5. **Real-world Applications** - Practical ML

---

*This guide will be your complete reference for understanding every aspect of these machine learning implementations.*
