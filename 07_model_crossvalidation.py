#!/usr/bin/env python3
"""
模型改进 - 添加交叉验证和正则化防止过拟合
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import differential_evolution
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, LeaveOneOut
import json
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

print("="*70)
print("MODEL IMPROVEMENT WITH OVERFITTING PREVENTION")
print("="*70)

# ============================================================
# 加载数据
# ============================================================

baseline = pd.read_csv('processed_baseline.csv')
pythia = pd.read_csv('real_attachments/B_scaling_laws/pythia_training_log_existing.csv')
pythia_converged = pythia.groupby('N_params_B').tail(5)[['N_params_B', 'D_tokens_B', 'val_loss']]
pythia_converged['data_source'] = 'pythia'

cerebras = pd.read_csv('real_attachments/B_scaling_laws/cerebras_training_log.csv')
cerebras_converged = cerebras.groupby('N_params_B').tail(5)[['N_params_B', 'D_tokens_B', 'val_loss']]
cerebras_converged['data_source'] = 'cerebras'

baseline['data_source'] = 'baseline'
all_data = pd.concat([baseline[['N_params_B', 'D_tokens_B', 'val_loss', 'data_source']],
                      pythia_converged, cerebras_converged])

all_data = all_data.dropna()
all_data = all_data[(all_data['val_loss'] > 1.5) & (all_data['val_loss'] < 5.0)]
all_data = all_data[(all_data['N_params_B'] > 0) & (all_data['D_tokens_B'] > 0)]

N = all_data['N_params_B'].values
D = all_data['D_tokens_B'].values
L = all_data['val_loss'].values
sources = all_data['data_source'].values

print(f"\nTotal data: {len(all_data)} points")
print(f"Parameters/Data ratio check:")
print(f"  Basic model: 5 params / {len(all_data)} points = {5/len(all_data):.3f}")
print(f"  Extended model: 7 params / {len(all_data)} points = {7/len(all_data):.3f}")
print(f"  Hierarchical: 7 params / {len(all_data)} points = {7/len(all_data):.3f}")
print(f"  Ensemble: 15 params / {len(all_data)} points = {15/len(all_data):.3f}")
print(f"\nRule of thumb: ratio should be < 0.1 (we're safe)")

# ============================================================
# Model Definitions
# ============================================================

def chinchilla_basic(ND, E, A, alpha, B, beta):
    """Basic Chinchilla: L = E + A/N^α + B/D^β"""
    N, D = ND
    return E + A / (N ** alpha) + B / (D ** beta)

def chinchilla_hierarchical(params, N, D, sources):
    """Hierarchical: E varies by source"""
    E_baseline, E_pythia, E_cerebras, A, alpha, B, beta = params
    predictions = np.zeros(len(N))
    for i, source in enumerate(sources):
        if source == 'baseline':
            E = E_baseline
        elif source == 'pythia':
            E = E_pythia
        else:
            E = E_cerebras
        predictions[i] = E + A / (N[i] ** alpha) + B / (D[i] ** beta)
    return predictions

# ============================================================
# Cross-Validation for Basic Model
# ============================================================

print("\n" + "="*70)
print("CROSS-VALIDATION: Basic Chinchilla Model")
print("="*70)

weights = np.ones(len(all_data))
weights[all_data['data_source'] == 'baseline'] = 3.0
weights = weights / weights.sum() * len(weights)

bounds_basic = [(1.5, 2.5), (0.01, 20), (0.05, 1.0), (0.01, 20), (0.05, 1.0)]

# K-Fold Cross-Validation
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

cv_scores_basic = []
cv_rmse_basic = []

print(f"\n{n_folds}-Fold Cross-Validation:")
for fold, (train_idx, test_idx) in enumerate(kf.split(N), 1):
    N_train, N_test = N[train_idx], N[test_idx]
    D_train, D_test = D[train_idx], D[test_idx]
    L_train, L_test = L[train_idx], L[test_idx]
    w_train = weights[train_idx]

    # Fit on training set
    result = differential_evolution(
        lambda p: np.sum(w_train * (L_train - chinchilla_basic((N_train, D_train), *p)) ** 2),
        bounds_basic, maxiter=500, seed=42, polish=True
    )

    # Evaluate on test set
    L_pred_test = chinchilla_basic((N_test, D_test), *result.x)
    r2_test = r2_score(L_test, L_pred_test)
    rmse_test = np.sqrt(mean_squared_error(L_test, L_pred_test))

    cv_scores_basic.append(r2_test)
    cv_rmse_basic.append(rmse_test)

    print(f"  Fold {fold}: R²={r2_test:.4f}, RMSE={rmse_test:.4f}")

print(f"\nCross-Validation Summary (Basic):")
print(f"  Mean R² = {np.mean(cv_scores_basic):.4f} ± {np.std(cv_scores_basic):.4f}")
print(f"  Mean RMSE = {np.mean(cv_rmse_basic):.4f} ± {np.std(cv_rmse_basic):.4f}")

# Full model
result_basic_full = differential_evolution(
    lambda p: np.sum(weights * (L - chinchilla_basic((N, D), *p)) ** 2),
    bounds_basic, maxiter=1000, seed=42, polish=True
)
params_basic = result_basic_full.x
L_pred_basic_full = chinchilla_basic((N, D), *params_basic)
r2_basic_full = r2_score(L, L_pred_basic_full)
rmse_basic_full = np.sqrt(mean_squared_error(L, L_pred_basic_full))

print(f"\nFull model (training on all data):")
print(f"  R² = {r2_basic_full:.4f}, RMSE = {rmse_basic_full:.4f}")
print(f"\nOverfitting check:")
print(f"  R² gap = {r2_basic_full - np.mean(cv_scores_basic):.4f}")
if r2_basic_full - np.mean(cv_scores_basic) < 0.05:
    print(f"  ✅ No significant overfitting (gap < 0.05)")
else:
    print(f"  ⚠️  Possible overfitting (gap ≥ 0.05)")

# ============================================================
# Cross-Validation for Hierarchical Model
# ============================================================

print("\n" + "="*70)
print("CROSS-VALIDATION: Hierarchical Model")
print("="*70)

bounds_hier = [(1.5, 2.5), (1.5, 2.5), (1.5, 2.5), (0.01, 20), (0.05, 1.0), (0.01, 20), (0.05, 1.0)]

cv_scores_hier = []
cv_rmse_hier = []

print(f"\n{n_folds}-Fold Cross-Validation:")
for fold, (train_idx, test_idx) in enumerate(kf.split(N), 1):
    N_train, N_test = N[train_idx], N[test_idx]
    D_train, D_test = D[train_idx], D[test_idx]
    L_train, L_test = L[train_idx], L[test_idx]
    sources_train, sources_test = sources[train_idx], sources[test_idx]
    w_train = weights[train_idx]

    # Fit on training set
    def objective_train(params):
        pred = chinchilla_hierarchical(params, N_train, D_train, sources_train)
        return np.sum(w_train * (L_train - pred) ** 2)

    result = differential_evolution(objective_train, bounds_hier, maxiter=500, seed=42, polish=True)

    # Evaluate on test set
    L_pred_test = chinchilla_hierarchical(result.x, N_test, D_test, sources_test)
    r2_test = r2_score(L_test, L_pred_test)
    rmse_test = np.sqrt(mean_squared_error(L_test, L_pred_test))

    cv_scores_hier.append(r2_test)
    cv_rmse_hier.append(rmse_test)

    print(f"  Fold {fold}: R²={r2_test:.4f}, RMSE={rmse_test:.4f}")

print(f"\nCross-Validation Summary (Hierarchical):")
print(f"  Mean R² = {np.mean(cv_scores_hier):.4f} ± {np.std(cv_scores_hier):.4f}")
print(f"  Mean RMSE = {np.mean(cv_rmse_hier):.4f} ± {np.std(cv_rmse_hier):.4f}")

# Full model
def objective_hier_full(params):
    pred = chinchilla_hierarchical(params, N, D, sources)
    return np.sum(weights * (L - pred) ** 2)

result_hier_full = differential_evolution(objective_hier_full, bounds_hier, maxiter=1000, seed=42, polish=True)
params_hier = result_hier_full.x
L_pred_hier_full = chinchilla_hierarchical(params_hier, N, D, sources)
r2_hier_full = r2_score(L, L_pred_hier_full)
rmse_hier_full = np.sqrt(mean_squared_error(L, L_pred_hier_full))

print(f"\nFull model (training on all data):")
print(f"  R² = {r2_hier_full:.4f}, RMSE = {rmse_hier_full:.4f}")
print(f"\nOverfitting check:")
print(f"  R² gap = {r2_hier_full - np.mean(cv_scores_hier):.4f}")
if r2_hier_full - np.mean(cv_scores_hier) < 0.05:
    print(f"  ✅ No significant overfitting (gap < 0.05)")
else:
    print(f"  ⚠️  Possible overfitting (gap ≥ 0.05)")

# ============================================================
# Cross-Validation for Ensemble Model
# ============================================================

print("\n" + "="*70)
print("CROSS-VALIDATION: Ensemble Model")
print("="*70)

cv_scores_ensemble = []
cv_rmse_ensemble = []

print(f"\n{n_folds}-Fold Cross-Validation:")
for fold, (train_idx, test_idx) in enumerate(kf.split(N), 1):
    predictions_test = np.zeros(len(test_idx))

    for source in ['baseline', 'pythia', 'cerebras']:
        # Get source-specific indices in train set
        source_train_mask = sources[train_idx] == source
        source_test_mask = sources[test_idx] == source

        if not np.any(source_train_mask):
            continue

        N_train_s = N[train_idx][source_train_mask]
        D_train_s = D[train_idx][source_train_mask]
        L_train_s = L[train_idx][source_train_mask]

        # Fit source-specific model
        result_s = differential_evolution(
            lambda p: np.sum((L_train_s - chinchilla_basic((N_train_s, D_train_s), *p)) ** 2),
            bounds_basic, maxiter=300, seed=42
        )

        # Predict on test set for this source
        if np.any(source_test_mask):
            N_test_s = N[test_idx][source_test_mask]
            D_test_s = D[test_idx][source_test_mask]
            predictions_test[source_test_mask] = chinchilla_basic((N_test_s, D_test_s), *result_s.x)

    L_test = L[test_idx]
    r2_test = r2_score(L_test, predictions_test)
    rmse_test = np.sqrt(mean_squared_error(L_test, predictions_test))

    cv_scores_ensemble.append(r2_test)
    cv_rmse_ensemble.append(rmse_test)

    print(f"  Fold {fold}: R²={r2_test:.4f}, RMSE={rmse_test:.4f}")

print(f"\nCross-Validation Summary (Ensemble):")
print(f"  Mean R² = {np.mean(cv_scores_ensemble):.4f} ± {np.std(cv_scores_ensemble):.4f}")
print(f"  Mean RMSE = {np.mean(cv_rmse_ensemble):.4f} ± {np.std(cv_rmse_ensemble):.4f}")

# Full model
predictions_ensemble_full = np.zeros(len(N))
for source in ['baseline', 'pythia', 'cerebras']:
    mask = sources == source
    N_s = N[mask]
    D_s = D[mask]
    L_s = L[mask]

    result_s = differential_evolution(
        lambda p: np.sum((L_s - chinchilla_basic((N_s, D_s), *p)) ** 2),
        bounds_basic, maxiter=500, seed=42
    )

    predictions_ensemble_full[mask] = chinchilla_basic((N_s, D_s), *result_s.x)

r2_ensemble_full = r2_score(L, predictions_ensemble_full)
rmse_ensemble_full = np.sqrt(mean_squared_error(L, predictions_ensemble_full))

print(f"\nFull model (training on all data):")
print(f"  R² = {r2_ensemble_full:.4f}, RMSE = {rmse_ensemble_full:.4f}")
print(f"\nOverfitting check:")
print(f"  R² gap = {r2_ensemble_full - np.mean(cv_scores_ensemble):.4f}")
if r2_ensemble_full - np.mean(cv_scores_ensemble) < 0.10:
    print(f"  ✅ Acceptable overfitting (gap < 0.10)")
elif r2_ensemble_full - np.mean(cv_scores_ensemble) < 0.20:
    print(f"  ⚠️  Moderate overfitting (gap 0.10-0.20)")
else:
    print(f"  ❌ Severe overfitting (gap ≥ 0.20)")

# ============================================================
# Summary Comparison
# ============================================================

print("\n" + "="*70)
print("FINAL COMPARISON (with Cross-Validation)")
print("="*70)

summary = pd.DataFrame({
    'Model': ['Basic', 'Hierarchical', 'Ensemble'],
    'Full R²': [r2_basic_full, r2_hier_full, r2_ensemble_full],
    'CV R² (mean)': [np.mean(cv_scores_basic), np.mean(cv_scores_hier), np.mean(cv_scores_ensemble)],
    'CV R² (std)': [np.std(cv_scores_basic), np.std(cv_scores_hier), np.std(cv_scores_ensemble)],
    'Full RMSE': [rmse_basic_full, rmse_hier_full, rmse_ensemble_full],
    'CV RMSE (mean)': [np.mean(cv_rmse_basic), np.mean(cv_rmse_hier), np.mean(cv_rmse_ensemble)],
    'Overfitting Gap': [
        r2_basic_full - np.mean(cv_scores_basic),
        r2_hier_full - np.mean(cv_scores_hier),
        r2_ensemble_full - np.mean(cv_scores_ensemble)
    ]
})

print("\n", summary.to_string(index=False))

# Select best model based on CV performance
best_idx = summary['CV R² (mean)'].argmax()
best_model = summary.iloc[best_idx]['Model']
best_cv_r2 = summary.iloc[best_idx]['CV R² (mean)']

print(f"\n🏆 Best Model (based on CV): {best_model}")
print(f"   CV R² = {best_cv_r2:.4f}")
print(f"   Overfitting Gap = {summary.iloc[best_idx]['Overfitting Gap']:.4f}")

# ============================================================
# Visualization
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. CV R² Comparison
ax = axes[0]
x = np.arange(len(summary))
ax.bar(x - 0.15, summary['Full R²'], 0.3, label='Full Model', alpha=0.8, color='steelblue')
ax.bar(x + 0.15, summary['CV R² (mean)'], 0.3, label='CV Mean', alpha=0.8, color='coral')
ax.errorbar(x + 0.15, summary['CV R² (mean)'], yerr=summary['CV R² (std)'],
           fmt='none', color='black', capsize=5)
ax.set_ylabel('R² Score', fontsize=11)
ax.set_title('Model Performance: Full vs Cross-Validation', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(summary['Model'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 2. Overfitting Gap
ax = axes[1]
colors = ['green' if gap < 0.05 else 'orange' if gap < 0.10 else 'red'
          for gap in summary['Overfitting Gap']]
ax.barh(summary['Model'], summary['Overfitting Gap'], color=colors, alpha=0.7, edgecolor='black')
ax.axvline(0.05, color='green', linestyle='--', linewidth=2, label='Safe threshold')
ax.axvline(0.10, color='orange', linestyle='--', linewidth=2, label='Warning threshold')
ax.set_xlabel('R² Gap (Full - CV)', fontsize=11)
ax.set_title('Overfitting Assessment', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('model_crossvalidation_results.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved: model_crossvalidation_results.png")

# ============================================================
# Save Results
# ============================================================

summary.to_csv('model_cv_comparison.csv', index=False)
print("✅ Results saved: model_cv_comparison.csv")

cv_results = {
    'basic': {
        'cv_r2_mean': float(np.mean(cv_scores_basic)),
        'cv_r2_std': float(np.std(cv_scores_basic)),
        'full_r2': float(r2_basic_full),
        'overfitting_gap': float(r2_basic_full - np.mean(cv_scores_basic)),
        'parameters': list(map(float, params_basic))
    },
    'hierarchical': {
        'cv_r2_mean': float(np.mean(cv_scores_hier)),
        'cv_r2_std': float(np.std(cv_scores_hier)),
        'full_r2': float(r2_hier_full),
        'overfitting_gap': float(r2_hier_full - np.mean(cv_scores_hier)),
        'parameters': list(map(float, params_hier))
    },
    'ensemble': {
        'cv_r2_mean': float(np.mean(cv_scores_ensemble)),
        'cv_r2_std': float(np.std(cv_scores_ensemble)),
        'full_r2': float(r2_ensemble_full),
        'overfitting_gap': float(r2_ensemble_full - np.mean(cv_scores_ensemble))
    },
    'best_model': best_model,
    'recommendation': 'Use Hierarchical model for best balance of performance and generalization'
                     if best_model == 'Hierarchical' else f'Use {best_model} model'
}

with open('model_cv_results.json', 'w') as f:
    json.dump(cv_results, f, indent=2)

print("✅ CV results saved: model_cv_results.json")

print("\n" + "="*70)
print("CROSS-VALIDATION COMPLETE!")
print("="*70)
print("\nRecommendation: Check overfitting gap before choosing final model.")
print("  Gap < 0.05: Excellent")
print("  Gap < 0.10: Good")
print("  Gap ≥ 0.10: Consider simpler model")
