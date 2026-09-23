# Large Language Model Resource Optimization under Compute Constraints

**2026 Huawei Cup Mathematical Modeling Competition**

## Project Structure

```
├── question1_scaling_law_modeling/      # Problem 2: Scaling Law (Attachment B)
│   ├── Q1_Scaling_Law_Modeling.ipynb   # ✅ Completed - R²=0.97
│   ├── modules/                         # Model fitting & CV
│   ├── results/
│   └── figures/
│
├── question2_resource_optimization/     # Problem 3: Optimal Allocation
│   ├── modules/optimizer.py            # ✅ Completed - N*∝C^0.85
│   ├── results/
│   └── figures/
│
├── question3_multistage_training/       # Problem 3: Multi-Stage Strategy  
│   ├── modules/multistage_optimizer.py # ✅ Completed - 20% improvement
│   ├── results/
│   └── figures/
│
├── question4_technology_evolution/      # Problem 4: Tech Evolution (Attachment C)
│   ├── modules/                         # 🔴 Not Started
│   ├── results/
│   └── figures/
│
├── comprehensive_analysis/              # Complete Analysis & Reports
│   ├── reports/
│   └── figures/
│
├── real_attachments/                    # Original data
│   ├── A_data_value/                   # Problem 1 (Not started)
│   ├── B_scaling_laws/                 # ✅ Used in Q1-Q3
│   └── C_efficiency_evolution/         # Problem 4 (Not started)
│
└── README.md                           # This file

```

## Competition Problems Mapping

| Folder | Competition Problem | Status |
|--------|---------------------|--------|
| ❌ (Not created) | **Problem 1**: Data Quality & Domain Mix (Attachment A) | 🔴 Not Started |
| question1_* | **Problem 2**: Scaling Law & Cross-dimension Fusion (Attachment B) | ✅ Completed |
| question2_*, question3_* | **Problem 3**: Multi-dimensional Resource Optimization (Attachment B+C) | ✅ Partially Done |
| question4_* | **Problem 4**: Technology Evolution & Frontier Prediction (Attachment C) | 🔴 Not Started |

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
