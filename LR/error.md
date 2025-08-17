# Regression Model Evaluation Metrics
A comprehensive guide to evaluating regression model performance with standard metrics and practical examples.

## Overview
When building regression models, we need to measure how well they predict continuous values. Unlike classification problems, traditional metrics like accuracy, precision, and recall are not meaningful for regression because predictions are continuous rather than discrete categories.

## Table of Contents

- [Standard Regression Metrics](#standard-regression-metrics)
  - [Mean Absolute Error (MAE)](#mean-absolute-error-mae)
  - [Mean Squared Error (MSE)](#mean-squared-error-mse)
  - [Root Mean Squared Error (RMSE)](#root-mean-squared-error-rmse)
  - [R² Score (Coefficient of Determination)](#r²-score-coefficient-of-determination)
- [Additional Metrics](#additional-metrics)
  - [Mean Absolute Percentage Error (MAPE)](#mean-absolute-percentage-error-mape)
  - [Adjusted R²](#adjusted-r²)
- [Custom Tolerance-Based Accuracy](#custom-tolerance-based-accuracy)
- [Implementation Examples](#implementation-examples)
- [Choosing the Right Metric](#choosing-the-right-metric)
- [Best Practices](#best-practices)

## Standard Regression Metrics

### Mean Absolute Error (MAE)

```
MAE = (1/n) × Σ |y_i - ŷ_i|
```

**What it measures**: The average absolute difference between predicted and actual values.

**Characteristics**:
- Easy to interpret: "On average, predictions are off by X units"
- Robust to outliers
- Linear penalty for errors
- Same units as the target variable

**Use when**: You want a straightforward, interpretable error metric that treats all errors equally.

### Mean Squared Error (MSE)

```
MSE = (1/n) × Σ (y_i - ŷ_i)²
```

**What it measures**: The average of squared differences between predicted and actual values.

**Characteristics**:
- Penalizes large errors more heavily (quadratic penalty)
- Always positive
- Units are squared compared to the target variable
- Sensitive to outliers

**Use when**: Large errors are significantly worse than small errors in your domain.

### Root Mean Squared Error (RMSE)

```
RMSE = √MSE = √[(1/n) × Σ (y_i - ŷ_i)²]
```

**What it measures**: The square root of MSE, bringing the metric back to the original units.

**Characteristics**:
- Same sensitivity to outliers as MSE
- Interpretable in the same units as the target
- Most commonly reported error metric
- Penalizes large errors more than MAE

**Use when**: You want MSE's outlier sensitivity but need interpretable units.

### R² Score (Coefficient of Determination)

```
R² = 1 - [Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²]
```

**What it measures**: The proportion of variance in the dependent variable explained by the model.

**Interpretation**:
- **R² = 1**: Perfect fit
- **R² = 0**: Model performs no better than predicting the mean
- **R² < 0**: Model performs worse than predicting the mean

**Characteristics**:
- Scale-independent
- Can be negative
- Popular for model comparison
- Can be misleading with non-linear relationships

## Additional Metrics

### Mean Absolute Percentage Error (MAPE)

```
MAPE = (1/n) × Σ |(y_i - ŷ_i) / y_i| × 100
```

**What it measures**: Average absolute percentage error between predicted and actual values.

**Characteristics**:
- Scale-independent (expressed as percentage)
- Easy to communicate to stakeholders
- Undefined when actual values are zero
- Biased toward predictions that are too low

### Adjusted R²

```
R²_adj = 1 - (1 - R²) × (n - 1) / (n - p - 1)
```

Where:
- n = number of observations
- p = number of predictors

**What it measures**: R² adjusted for the number of predictors in the model.

**Characteristics**:
- Penalizes addition of irrelevant features
- Always less than or equal to R²
- Better for model comparison with different numbers of features

## Custom Tolerance-Based Accuracy

For business contexts where "close enough" predictions have practical value:

```python
def regression_accuracy(y_true, y_pred, tolerance=0.5):
    """
    Calculate accuracy for regression based on tolerance threshold.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        tolerance: Acceptable error threshold
    
    Returns:
        Proportion of predictions within tolerance
    """
    correct = np.abs(np.array(y_true) - np.array(y_pred)) <= tolerance
    return np.mean(correct)
```

## Implementation Examples

### Python with scikit-learn

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Sample data
y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]

# Calculate metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print(f"MAE: {mae:.3f}")
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")

# Custom tolerance-based accuracy
tolerance = 0.5
accuracy = regression_accuracy(y_true, y_pred, tolerance)
print(f"Custom Accuracy (±{tolerance}): {accuracy:.1%}")
```

### Complete Evaluation Function

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_regression(y_true, y_pred, tolerance=None):
    """
    Comprehensive regression evaluation.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        tolerance: Optional tolerance for custom accuracy
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R²': r2_score(y_true, y_pred)
    }
    
    # Add MAPE if no zero values
    if not np.any(np.array(y_true) == 0):
        mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100
        metrics['MAPE'] = mape
    
    # Add custom accuracy if tolerance specified
    if tolerance is not None:
        accuracy = regression_accuracy(y_true, y_pred, tolerance)
        metrics[f'Accuracy_±{tolerance}'] = accuracy
    
    return metrics
```

## Choosing the Right Metric

| Metric | Best Used When | Avoid When |
|--------|----------------|------------|
| MAE | Want interpretable, robust metric | Large errors are much worse than small ones |
| RMSE | Large errors are costly; need interpretable units | Data has many outliers |
| R² | Comparing models; understanding variance explained | Non-linear relationships; extrapolating |
| MAPE | Need scale-independent percentage errors | Target values can be zero |

## Best Practices

### 1. Use Multiple Metrics
No single metric tells the complete story. Always evaluate with at least 2-3 different metrics.

### 2. Consider Your Domain
- **Finance**: Large errors might be catastrophic → Use RMSE
- **Marketing**: Consistent performance matters → Use MAE
- **Science**: Understanding relationship strength → Use R²

### 3. Validate Across Data Splits
Ensure metrics are stable using cross-validation or train/validation/test splits.

### 4. Visualize Residuals
Create residual plots to identify patterns that metrics might miss:

```python
import matplotlib.pyplot as plt

def plot_residuals(y_true, y_pred):
    residuals = np.array(y_true) - np.array(y_pred)
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    
    plt.tight_layout()
    plt.show()
```

### 5. Context-Specific Metrics
Consider creating domain-specific metrics that align with business objectives:

```python
def business_metric(y_true, y_pred, cost_underestimate=10, cost_overestimate=1):
    """
    Custom metric where underestimating costs more than overestimating.
    """
    errors = np.array(y_true) - np.array(y_pred)
    costs = np.where(errors > 0, 
                    errors * cost_underestimate,  # Underestimate penalty
                    -errors * cost_overestimate)  # Overestimate penalty
    return np.mean(costs)
```

## Contributing

Feel free to contribute additional metrics, examples, or improvements to this guide.

## License

This documentation is provided under the MIT License.