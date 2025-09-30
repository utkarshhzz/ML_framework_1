# 📁 Datasets Directory
*Sample Datasets and Data Information for ML Framework*

## 🎯 Overview

This directory contains sample datasets and comprehensive information about all datasets used in our ML framework. Each dataset includes detailed explanations of features, target variables, and practical use cases.

---

## 📊 Available Datasets

### 🔬 Built-in Scikit-learn Datasets

#### 1. Iris Dataset (Classification)
**Location**: `sklearn.datasets.load_iris()`

**Description**: Classic dataset of iris flower measurements for species classification.

**Features (4):**
- `sepal_length`: Length of sepal in cm
- `sepal_width`: Width of sepal in cm  
- `petal_length`: Length of petal in cm
- `petal_width`: Width of petal in cm

**Target**: Species classification (3 classes)
- `0`: Iris Setosa
- `1`: Iris Versicolor  
- `2`: Iris Virginica

**Dataset Characteristics:**
- **Samples**: 150 (50 per class)
- **Features**: 4 numerical
- **Classes**: 3 (perfectly balanced)
- **Missing values**: None
- **Difficulty**: Easy (linearly separable)

**Best for learning:**
- Multi-class classification
- Feature visualization
- Algorithm comparison
- Basic ML concepts

**Example code:**
```python
from sklearn.datasets import load_iris
data = load_iris()
X, y = data.data, data.target
feature_names = data.feature_names
target_names = data.target_names
```

---

#### 2. Wine Dataset (Classification)
**Location**: `sklearn.datasets.load_wine()`

**Description**: Wine chemical analysis for origin classification.

**Features (13):**
- `alcohol`: Alcohol content
- `malic_acid`: Malic acid content
- `ash`: Ash content
- `alcalinity_of_ash`: Alcalinity of ash
- And 9 more chemical measurements...

**Target**: Wine origin (3 classes)
- `0`: Class 0 wines
- `1`: Class 1 wines
- `2`: Class 2 wines

**Dataset Characteristics:**
- **Samples**: 178
- **Features**: 13 numerical
- **Classes**: 3 (imbalanced)
- **Missing values**: None
- **Difficulty**: Medium

**Best for learning:**
- Feature scaling importance
- High-dimensional classification
- Imbalanced classes handling

---

#### 3. Breast Cancer Dataset (Binary Classification)
**Location**: `sklearn.datasets.load_breast_cancer()`

**Description**: Breast cancer diagnosis from cell nuclei measurements.

**Features (30)**: Various measurements of cell nuclei
- `mean_radius`: Mean radius of nuclei
- `mean_texture`: Mean texture
- `mean_perimeter`: Mean perimeter
- And 27 more measurements...

**Target**: Diagnosis (2 classes)
- `0`: Malignant (cancerous)
- `1`: Benign (non-cancerous)

**Dataset Characteristics:**
- **Samples**: 569
- **Features**: 30 numerical
- **Classes**: 2 (imbalanced: ~63% benign)
- **Missing values**: None
- **Difficulty**: Medium

**Best for learning:**
- Binary classification
- Medical diagnosis scenarios
- High-dimensional data
- Feature importance analysis

---

#### 4. Digits Dataset (Multi-class Classification)
**Location**: `sklearn.datasets.load_digits()`

**Description**: Handwritten digit recognition (8x8 pixel images).

**Features (64)**: Pixel intensities (0-16) in 8x8 grid

**Target**: Digit classification (10 classes)
- `0-9`: Handwritten digits

**Dataset Characteristics:**
- **Samples**: 1,797
- **Features**: 64 numerical (pixel values)
- **Classes**: 10 (roughly balanced)
- **Missing values**: None
- **Difficulty**: Medium

**Best for learning:**
- Image classification basics
- Multi-class problems
- Dimensionality reduction
- Visualization techniques

---

#### 5. Boston Housing Dataset (Regression)
**Location**: `sklearn.datasets.load_boston()` (deprecated - use California housing)

**Note**: Deprecated due to ethical concerns. Use California housing instead.

---

#### 6. California Housing Dataset (Regression)
**Location**: `sklearn.datasets.fetch_california_housing()`

**Description**: Housing prices in California districts.

**Features (8):**
- `MedInc`: Median income in block group
- `HouseAge`: Median house age in block group
- `AveRooms`: Average number of rooms per household
- `AveBedrms`: Average number of bedrooms per household
- `Population`: Block group population
- `AveOccup`: Average number of household members
- `Latitude`: Block group latitude
- `Longitude`: Block group longitude

**Target**: Median house value (in hundreds of thousands of dollars)

**Dataset Characteristics:**
- **Samples**: 20,640
- **Features**: 8 numerical
- **Target**: Continuous (regression)
- **Missing values**: None
- **Difficulty**: Medium

**Best for learning:**
- Regression problems
- Feature engineering
- Geographic data
- Large dataset handling

---

#### 7. Diabetes Dataset (Regression)
**Location**: `sklearn.datasets.load_diabetes()`

**Description**: Diabetes progression prediction from patient data.

**Features (10)**: Patient measurements (age, sex, BMI, blood pressure, and 6 serum measurements)

**Target**: Disease progression (continuous value)

**Dataset Characteristics:**
- **Samples**: 442
- **Features**: 10 numerical (standardized)
- **Target**: Continuous
- **Missing values**: None
- **Difficulty**: Medium

**Best for learning:**
- Small dataset regression
- Medical prediction
- Feature preprocessing effects
- Overfitting detection

---

### 🎲 Synthetic Datasets

#### 1. Make Classification
**Function**: `sklearn.datasets.make_classification()`

**Purpose**: Generate synthetic classification datasets with controlled properties.

**Key Parameters:**
- `n_samples`: Number of samples
- `n_features`: Number of features
- `n_classes`: Number of classes
- `n_informative`: Number of informative features
- `n_redundant`: Number of redundant features
- `class_sep`: Class separation (difficulty)

**Example:**
```python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=3,
    class_sep=1.0,
    random_state=42
)
```

**Best for learning:**
- Algorithm comparison
- Controlled experiments
- Understanding feature importance
- Testing preprocessing methods

---

#### 2. Make Regression  
**Function**: `sklearn.datasets.make_regression()`

**Purpose**: Generate synthetic regression datasets.

**Key Parameters:**
- `n_samples`: Number of samples
- `n_features`: Number of features
- `noise`: Standard deviation of noise
- `n_informative`: Number of informative features

**Example:**
```python
from sklearn.datasets import make_regression

X, y = make_regression(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    noise=0.1,
    random_state=42
)
```

---

#### 3. Make Blobs
**Function**: `sklearn.datasets.make_blobs()`

**Purpose**: Generate Gaussian blobs for clustering and classification.

**Example:**
```python
from sklearn.datasets import make_blobs

X, y = make_blobs(
    n_samples=300,
    centers=4,
    n_features=2,
    cluster_std=1.0,
    random_state=42
)
```

**Best for learning:**
- Clustering algorithms
- 2D visualization
- Decision boundary visualization
- Simple classification

---

## 🎯 Dataset Usage Guidelines

### 📊 Choosing the Right Dataset

**For Beginners:**
1. **Iris** - Start here for classification
2. **Make_blobs** - Visual understanding of clustering
3. **Diabetes** - Small regression problem

**For Intermediate:**
1. **Wine** - Multi-feature classification
2. **California Housing** - Realistic regression
3. **Breast Cancer** - Important binary classification

**For Advanced:**
1. **Digits** - Image-like data
2. **Custom synthetic** - Controlled experiments
3. **Real-world external** - Domain-specific challenges

### 🔍 Dataset Analysis Checklist

When working with any dataset, always examine:

1. **Basic Properties:**
   ```python
   print(f"Shape: {X.shape}")
   print(f"Features: {data.feature_names}")
   print(f"Target: {data.target_names}")
   ```

2. **Missing Values:**
   ```python
   print(f"Missing values: {pd.DataFrame(X).isnull().sum().sum()}")
   ```

3. **Class Distribution (Classification):**
   ```python
   unique, counts = np.unique(y, return_counts=True)
   print(f"Class distribution: {dict(zip(unique, counts))}")
   ```

4. **Target Distribution (Regression):**
   ```python
   print(f"Target stats: mean={y.mean():.2f}, std={y.std():.2f}")
   ```

5. **Feature Scales:**
   ```python
   print(f"Feature ranges:")
   for i, name in enumerate(feature_names):
       print(f"{name}: [{X[:, i].min():.2f}, {X[:, i].max():.2f}]")
   ```

---

## 🛠️ Data Loading Utilities

### 📊 Standard Loading Function

```python
def load_dataset(dataset_name, return_df=False):
    """
    Load dataset with comprehensive information display.
    
    Parameters:
    - dataset_name: str, name of dataset to load
    - return_df: bool, whether to return pandas DataFrame
    
    Returns:
    - X, y, feature_names, target_names (and optionally DataFrames)
    """
    
    # Dataset loading dictionary
    datasets = {
        'iris': load_iris,
        'wine': load_wine,
        'breast_cancer': load_breast_cancer,
        'digits': load_digits,
        'diabetes': load_diabetes,
        'california_housing': fetch_california_housing
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Dataset {dataset_name} not available")
    
    # Load dataset
    data = datasets[dataset_name]()
    X, y = data.data, data.target
    
    # Get names
    feature_names = getattr(data, 'feature_names', [f'feature_{i}' for i in range(X.shape[1])])
    target_names = getattr(data, 'target_names', None)
    
    # Display basic info
    print(f"📊 {dataset_name.title()} Dataset Loaded")
    print(f"Shape: {X.shape}")
    print(f"Features: {len(feature_names)}")
    print(f"Samples: {len(y)}")
    
    if len(np.unique(y)) < 20:  # Classification
        unique, counts = np.unique(y, return_counts=True)
        print(f"Classes: {len(unique)}")
        print(f"Distribution: {dict(zip(unique, counts))}")
    else:  # Regression
        print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    if return_df:
        df_X = pd.DataFrame(X, columns=feature_names)
        df_y = pd.Series(y, name='target')
        return X, y, feature_names, target_names, df_X, df_y
    
    return X, y, feature_names, target_names
```

### 🎯 Example Usage

```python
# Load iris dataset
X, y, feature_names, target_names = load_dataset('iris')

# Load with pandas DataFrames
X, y, feature_names, target_names, df_X, df_y = load_dataset('iris', return_df=True)

# Quick exploratory analysis
from visualization_tools import quick_eda
quick_eda(df_X.assign(target=df_y), 'target')
```

---

## 📈 Dataset Preprocessing Examples

### 🎯 Standard Preprocessing Pipeline

```python
from data_preprocessing import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Load dataset
X, y, feature_names, target_names = load_dataset('breast_cancer')

# Create DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Run full preprocessing pipeline
X_train, X_test, y_train, y_test = preprocessor.full_preprocessing_pipeline(
    df, 'target', test_size=0.2
)

print("Data ready for modeling!")
```

---

## 🎯 External Dataset Integration

### 📊 Adding Your Own Datasets

1. **Place CSV files in `sample_datasets/` directory**
2. **Document the dataset** in this README
3. **Use standard loading pattern**:

```python
def load_custom_dataset(filepath):
    """Load custom dataset with standard format."""
    df = pd.read_csv(filepath)
    
    # Assume last column is target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    feature_names = df.columns[:-1].tolist()
    
    return X, y, feature_names
```

### 🌐 Popular External Datasets

**Kaggle Datasets (require download):**
- Titanic (Classification)
- House Prices (Regression)  
- Credit Card Fraud (Imbalanced Classification)

**UCI ML Repository:**
- Adult Income (Classification)
- Abalone (Regression)
- Mushroom (Classification)

---

## 🎯 Best Practices

### ✅ Do's
1. **Always explore data first** - Use EDA before modeling
2. **Check for missing values** - Handle appropriately
3. **Understand your target** - Classification vs regression
4. **Split data properly** - Use stratification for classification
5. **Document data sources** - Keep track of where data comes from
6. **Validate data quality** - Check for outliers and errors

### ❌ Don'ts
1. **Don't skip exploratory analysis** - Understand your data first
2. **Don't ignore class imbalance** - Address if present
3. **Don't forget to scale features** - Especially for SVM and neural networks
4. **Don't use test data for preprocessing** - Fit only on training data
5. **Don't assume data is clean** - Always validate quality

---

## 📚 Learning Resources

### 🎯 Dataset-Specific Tutorials
Each algorithm notebook in our framework includes:
- Dataset loading and exploration
- Appropriate preprocessing steps
- Algorithm application
- Results interpretation

### 📖 External Resources
- [UCI ML Repository](https://archive.ics.uci.edu/ml/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Scikit-learn Datasets](https://scikit-learn.org/stable/datasets.html)
- [OpenML](https://www.openml.org/)

---

*This datasets guide helps you understand and effectively use all datasets in our ML framework. Combine with the preprocessing utilities and visualization tools for complete data analysis workflows.*