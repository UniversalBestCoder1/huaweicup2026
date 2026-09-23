# Large Language Model Resource Optimization under Compute Constraints

**2026 Huawei Cup Mathematical Modeling Competition**

## Project Structure

```
├── question1_scaling_law_modeling/      # Problem 1: Scaling Law
│   ├── Q1_Scaling_Law_Modeling.ipynb
│   ├── modules/
│   ├── results/
│   └── figures/
│
├── question2_resource_optimization/     # Problem 2: Optimal Allocation
│   ├── Q2_Resource_Optimization.ipynb
│   ├── modules/
│   ├── results/
│   └── figures/
│
├── question3_multistage_training/       # Problem 3: Multi-Stage Strategy
│   ├── Q3_MultiStage_Training.ipynb
│   ├── modules/
│   ├── results/
│   └── figures/
│
├── comprehensive_analysis/              # Complete Analysis & Reports
│   ├── reports/
│   └── figures/
│
├── real_attachments/                    # Original data
│   └── B_scaling_laws/
│
└── README.md                           # This file

```

## Quick Start

### Question 1: Scaling Law Modeling
```bash
cd question1_scaling_law_modeling
jupyter notebook Q1_Scaling_Law_Modeling.ipynb
```

### Question 2: Resource Optimization
```bash
cd question2_resource_optimization
jupyter notebook Q2_Resource_Optimization.ipynb
```

### Question 3: Multi-Stage Training
```bash
cd question3_multistage_training
jupyter notebook Q3_MultiStage_Training.ipynb
```

## Key Results

| Problem | Key Finding | Metric |
|---------|-------------|--------|
| Q1: Scaling Law | Hierarchical model with CV | R²=0.97 (CV: 0.96) |
| Q2: Optimization | N* ∝ C^0.85, D* ∝ C^0.15 | Model-dominated |
| Q3: Multi-Stage | Expand strategy effective | ~20% improvement |

## Documentation

See `comprehensive_analysis/reports/` for detailed documentation.

## Data

Original data in `real_attachments/B_scaling_laws/`:
- 57 baseline convergence points
- Pythia training trajectories (8 models)
- Cerebras-GPT training logs

## Authors

- Mathematical modeling and analysis
- Implementation with Claude Opus 4.8

## Date

September 23, 2026
