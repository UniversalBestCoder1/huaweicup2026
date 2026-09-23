# Question 2: Resource Optimization

## Problem Statement
Given compute budget C, find optimal allocation (N*, D*) that minimizes loss.

## Structure

### Notebooks
- `Q2_Resource_Optimization.ipynb` - Optimization analysis

### Modules
- `modules/optimizer.py` - Resource allocation optimizer

### Results
- `results/optimal_allocations.csv` - Optimal N*, D* for various budgets

### Figures
- `figures/fig1_optimal_allocation.png` - Optimization results

## Key Results
- **Scaling**: N* ∝ C^0.85, D* ∝ C^0.15
- **Model size dominates resource allocation**
