# Question 1: Scaling Law Modeling

## Problem Statement
Establish a mathematical model (Scaling Law) describing the relationship between:
- Model size N (parameters)
- Training data D (tokens)
- Validation loss L

## Structure

### Notebooks
- `Q1_Scaling_Law_Modeling.ipynb` - Complete analysis workflow

### Modules
- `modules/data_exploration.py` - Data loading and EDA
- `modules/model_fitting.py` - Scaling law model fitting
- `modules/model_comparison.py` - Compare different model variants
- `modules/cross_validation.py` - Overfitting prevention

### Results
- `results/scaling_law_parameters.json` - Final fitted parameters
- `results/model_comparison.csv` - Model performance comparison
- `results/cv_results.json` - Cross-validation results

### Figures
- `figures/fig1_data_exploration.png` - Exploratory data analysis
- `figures/fig2_model_fitting.png` - Model fit visualization
- `figures/fig3_model_comparison.png` - Model comparison
- `figures/fig4_cross_validation.png` - CV results

## Key Results
- **Hierarchical Chinchilla Model**: R² = 0.97 (CV: 0.96±0.02)
- **No overfitting**: CV gap < 0.05
- **Source-specific intercepts with unified scaling laws**
