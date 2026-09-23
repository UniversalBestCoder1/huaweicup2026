# Model Improvement Plan - Iteration Log

## Current Model Performance

### Baseline (Chinchilla Scaling Law)
```
L(N, D) = E + A/N^α + B/D^β

Parameters:
- E = 1.5000
- A = 1.0622, α = 0.1736
- B = 7.3044, β = 1.0000
- R² = 0.5117, RMSE = 0.3247
```

### Key Issues Identified

1. **Low R² (0.51)** - Only explains 51% of variance
2. **Non-normal residuals** - Skewness = 0.63, p < 0.001
3. **Systematic bias by source**:
   - Cerebras: +0.519 (overestimated)
   - Pythia: -0.272 (underestimated)
   - Baseline: -0.047 (good)
4. **Correlation with D** - residuals corr(D) = 0.34

## Improvement Strategies

### Strategy 1: Robust Regression (Huber Loss)
**Goal**: Reduce impact of outliers and heterogeneous data sources
**Expected improvement**: +5-10% R²

### Strategy 2: Extended Model with Interaction Term
**Goal**: Capture N-D interaction effects
```
L(N, D) = E + A/N^α + B/D^β + C/(ND)^γ
```
**Expected improvement**: +10-15% R²

### Strategy 3: Hierarchical/Mixed-Effects Model
**Goal**: Account for data source heterogeneity
```
L(N, D) = E_source + A/N^α + B/D^β
where E_source varies by model family
```
**Expected improvement**: +15-20% R²

### Strategy 4: Non-parametric Model (Gaussian Process Regression)
**Goal**: Maximum flexibility without strong assumptions
**Expected improvement**: +20-30% R² (but less interpretable)

### Strategy 5: Ensemble Approach
**Goal**: Combine multiple models
- Fit separate models for each data source
- Weight by source confidence
**Expected improvement**: +15-25% R²

## Implementation Priority

1. **Priority 1**: Robust regression (quick, interpretable)
2. **Priority 2**: Extended model with interaction (still interpretable)
3. **Priority 3**: Hierarchical model (accounts for heterogeneity)
4. **Priority 4**: Ensemble (practical, good performance)
5. **Priority 5**: GPR (maximum performance, less interpretable)

## Next Steps

Starting with Priority 1 & 2 for balance of performance and interpretability.
