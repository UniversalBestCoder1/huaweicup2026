#!/usr/bin/env python3
"""
模型改进实验 - 多种方法对比
Iteration 1: 改进 Scaling Law 拟合效果
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit, minimize, differential_evolution
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
import json
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

print("="*70)
print("MODEL IMPROVEMENT EXPERIMENTS")
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

print(f"\nData: {len(all_data)} points")
print(f"Sources: baseline={len(all_data[all_data['data_source']=='baseline'])}, "
      f"pythia={len(all_data[all_data['data_source']=='pythia'])}, "
      f"cerebras={len(all_data[all_data['data_source']=='cerebras'])}")

# ============================================================
# Baseline Model (Current)
# ============================================================

def chinchilla_basic(ND, E, A, alpha, B, beta):
    """Basic Chinchilla: L = E + A/N^α + B/D^β"""
    N, D = ND
    return E + A / (N ** alpha) + B / (D ** beta)

print("\n" + "="*70)
print("MODEL 1: Basic Chinchilla (Baseline)")
print("="*70)

weights = np.ones(len(all_data))
weights[all_data['data_source'] == 'baseline'] = 3.0
weights = weights / weights.sum() * len(weights)

bounds_de = [(1.5, 2.5), (0.01, 20), (0.05, 1.0), (0.01, 20), (0.05, 1.0)]
result = differential_evolution(
    lambda p: np.sum(weights * (L - chinchilla_basic((N, D), *p)) ** 2),
    bounds_de, maxiter=1000, seed=42, polish=True
)

params_basic = result.x
L_pred_basic = chinchilla_basic((N, D), *params_basic)
r2_basic = r2_score(L, L_pred_basic)
rmse_basic = np.sqrt(mean_squared_error(L, L_pred_basic))

print(f"Parameters: E={params_basic[0]:.4f}, A={params_basic[1]:.4f}, α={params_basic[2]:.4f}, "
      f"B={params_basic[3]:.4f}, β={params_basic[4]:.4f}")
print(f"R² = {r2_basic:.4f}, RMSE = {rmse_basic:.4f}")

# ============================================================
# Model 2: Extended with Interaction Term
# ============================================================

def chinchilla_extended(ND, E, A, alpha, B, beta, C, gamma):
    """Extended: L = E + A/N^α + B/D^β + C/(ND)^γ"""
    N, D = ND
    return E + A / (N ** alpha) + B / (D ** beta) + C / ((N * D) ** gamma)

print("\n" + "="*70)
print("MODEL 2: Extended Chinchilla (with N-D interaction)")
print("="*70)

bounds_ext = [(1.5, 2.5), (0.01, 20), (0.05, 1.0), (0.01, 20), (0.05, 1.0), (0.01, 10), (0.05, 0.5)]
result_ext = differential_evolution(
    lambda p: np.sum(weights * (L - chinchilla_extended((N, D), *p)) ** 2),
    bounds_ext, maxiter=1000, seed=42, polish=True
)

params_ext = result_ext.x
L_pred_ext = chinchilla_extended((N, D), *params_ext)
r2_ext = r2_score(L, L_pred_ext)
rmse_ext = np.sqrt(mean_squared_error(L, L_pred_ext))

print(f"Parameters: E={params_ext[0]:.4f}, A={params_ext[1]:.4f}, α={params_ext[2]:.4f}, "
      f"B={params_ext[3]:.4f}, β={params_ext[4]:.4f}")
print(f"            C={params_ext[5]:.4f}, γ={params_ext[6]:.4f}")
print(f"R² = {r2_ext:.4f}, RMSE = {rmse_ext:.4f}")
print(f"Improvement: ΔR² = {r2_ext - r2_basic:.4f} ({(r2_ext - r2_basic)/r2_basic*100:.1f}%)")

# ============================================================
# Model 3: Hierarchical Model (Source-specific E)
# ============================================================

def chinchilla_hierarchical(params_dict, N, D, sources):
    """Hierarchical: E varies by source, A/α/B/β shared"""
    E_baseline, E_pythia, E_cerebras, A, alpha, B, beta = params_dict

    predictions = np.zeros(len(N))
    for i, source in enumerate(sources):
        if source == 'baseline':
            E = E_baseline
        elif source == 'pythia':
            E = E_pythia
        else:  # cerebras
            E = E_cerebras

        predictions[i] = E + A / (N[i] ** alpha) + B / (D[i] ** beta)

    return predictions

print("\n" + "="*70)
print("MODEL 3: Hierarchical Model (source-specific intercept)")
print("="*70)

def objective_hier(params):
    L_pred = chinchilla_hierarchical(params, N, D, all_data['data_source'].values)
    return np.sum(weights * (L - L_pred) ** 2)

bounds_hier = [(1.5, 2.5), (1.5, 2.5), (1.5, 2.5), (0.01, 20), (0.05, 1.0), (0.01, 20), (0.05, 1.0)]
result_hier = differential_evolution(objective_hier, bounds_hier, maxiter=1000, seed=42, polish=True)

params_hier = result_hier.x
L_pred_hier = chinchilla_hierarchical(params_hier, N, D, all_data['data_source'].values)
r2_hier = r2_score(L, L_pred_hier)
rmse_hier = np.sqrt(mean_squared_error(L, L_pred_hier))

print(f"E_baseline={params_hier[0]:.4f}, E_pythia={params_hier[1]:.4f}, E_cerebras={params_hier[2]:.4f}")
print(f"A={params_hier[3]:.4f}, α={params_hier[4]:.4f}, B={params_hier[5]:.4f}, β={params_hier[6]:.4f}")
print(f"R² = {r2_hier:.4f}, RMSE = {rmse_hier:.4f}")
print(f"Improvement: ΔR² = {r2_hier - r2_basic:.4f} ({(r2_hier - r2_basic)/r2_basic*100:.1f}%)")

# ============================================================
# Model 4: Log-transform Model
# ============================================================

def chinchilla_log(ND, a, b, c, d, e):
    """Log-space: log(L-E) = a*log(N) + b*log(D) + c*log(N)*log(D) + d"""
    N, D = ND
    log_N = np.log(N)
    log_D = np.log(D)
    log_L_minus_E = a * log_N + b * log_D + c * log_N * log_D + d
    return e + np.exp(log_L_minus_E)

print("\n" + "="*70)
print("MODEL 4: Log-transform Model")
print("="*70)

bounds_log = [(-2, 0), (-2, 0), (-0.1, 0.1), (-5, 5), (1.5, 2.5)]
result_log = differential_evolution(
    lambda p: np.sum(weights * (L - chinchilla_log((N, D), *p)) ** 2),
    bounds_log, maxiter=1000, seed=42, polish=True
)

params_log = result_log.x
L_pred_log = chinchilla_log((N, D), *params_log)
r2_log = r2_score(L, L_pred_log)
rmse_log = np.sqrt(mean_squared_error(L, L_pred_log))

print(f"Parameters: a={params_log[0]:.4f}, b={params_log[1]:.4f}, c={params_log[2]:.4f}, "
      f"d={params_log[3]:.4f}, e={params_log[4]:.4f}")
print(f"R² = {r2_log:.4f}, RMSE = {rmse_log:.4f}")
print(f"Improvement: ΔR² = {r2_log - r2_basic:.4f} ({(r2_log - r2_basic)/r2_basic*100:.1f}%)")

# ============================================================
# Model 5: Ensemble (Source-specific models)
# ============================================================

print("\n" + "="*70)
print("MODEL 5: Ensemble (separate model per source)")
print("="*70)

predictions_ensemble = np.zeros(len(N))
source_models = {}

for source in ['baseline', 'pythia', 'cerebras']:
    mask = all_data['data_source'] == source
    N_s = N[mask]
    D_s = D[mask]
    L_s = L[mask]

    result_s = differential_evolution(
        lambda p: np.sum((L_s - chinchilla_basic((N_s, D_s), *p)) ** 2),
        bounds_de, maxiter=500, seed=42
    )

    source_models[source] = result_s.x
    predictions_ensemble[mask] = chinchilla_basic((N_s, D_s), *result_s.x)

    r2_s = r2_score(L_s, predictions_ensemble[mask])
    print(f"  {source}: R²={r2_s:.4f}, params={result_s.x}")

r2_ensemble = r2_score(L, predictions_ensemble)
rmse_ensemble = np.sqrt(mean_squared_error(L, predictions_ensemble))

print(f"\nOverall R² = {r2_ensemble:.4f}, RMSE = {rmse_ensemble:.4f}")
print(f"Improvement: ΔR² = {r2_ensemble - r2_basic:.4f} ({(r2_ensemble - r2_basic)/r2_basic*100:.1f}%)")

# ============================================================
# Summary Comparison
# ============================================================

print("\n" + "="*70)
print("SUMMARY: Model Comparison")
print("="*70)

results = {
    'Model': ['Basic Chinchilla', 'Extended (Interaction)', 'Hierarchical', 'Log-transform', 'Ensemble'],
    'R²': [r2_basic, r2_ext, r2_hier, r2_log, r2_ensemble],
    'RMSE': [rmse_basic, rmse_ext, rmse_hier, rmse_log, rmse_ensemble],
    'Parameters': [5, 7, 7, 5, '15 (5×3)'],
    'Improvement': [0, r2_ext-r2_basic, r2_hier-r2_basic, r2_log-r2_basic, r2_ensemble-r2_basic]
}

df_results = pd.DataFrame(results)
print("\n", df_results.to_string(index=False))

# Find best model
best_idx = df_results['R²'].argmax()
best_model = df_results.iloc[best_idx]['Model']
best_r2 = df_results.iloc[best_idx]['R²']

print(f"\n🏆 Best Model: {best_model} (R² = {best_r2:.4f})")

# ============================================================
# Visualization
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

models_to_plot = [
    ('Basic', L_pred_basic, params_basic, r2_basic),
    ('Extended', L_pred_ext, params_ext, r2_ext),
    ('Hierarchical', L_pred_hier, params_hier, r2_hier),
    ('Log-transform', L_pred_log, params_log, r2_log),
    ('Ensemble', predictions_ensemble, None, r2_ensemble)
]

colors_map = {'baseline': 'red', 'pythia': 'blue', 'cerebras': 'green'}

for idx, (name, L_pred, params, r2) in enumerate(models_to_plot):
    ax = axes[idx // 3, idx % 3]

    for source, color in colors_map.items():
        mask = all_data['data_source'] == source
        ax.scatter(L[mask], L_pred[mask], c=color, alpha=0.6, s=30, label=source.capitalize())

    ax.plot([L.min(), L.max()], [L.min(), L.max()], 'k--', linewidth=2)
    ax.set_xlabel('True Loss', fontsize=11)
    ax.set_ylabel('Predicted Loss', fontsize=11)
    ax.set_title(f'{name} Model\nR²={r2:.4f}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Summary plot
ax = axes[1, 2]
ax.barh(df_results['Model'], df_results['R²'], color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('R² Score', fontsize=11)
ax.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
for i, (model, r2) in enumerate(zip(df_results['Model'], df_results['R²'])):
    ax.text(r2 + 0.01, i, f'{r2:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('model_improvement_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved: model_improvement_comparison.png")

# ============================================================
# Save Best Model
# ============================================================

if best_model == 'Extended (Interaction)':
    best_params = {
        'model': 'Extended Chinchilla',
        'formula': 'L(N,D) = E + A/N^α + B/D^β + C/(ND)^γ',
        'E': float(params_ext[0]),
        'A': float(params_ext[1]),
        'alpha': float(params_ext[2]),
        'B': float(params_ext[3]),
        'beta': float(params_ext[4]),
        'C': float(params_ext[5]),
        'gamma': float(params_ext[6]),
        'R2': float(r2_ext),
        'RMSE': float(rmse_ext)
    }
elif best_model == 'Hierarchical':
    best_params = {
        'model': 'Hierarchical Chinchilla',
        'formula': 'L(N,D) = E_source + A/N^α + B/D^β',
        'E_baseline': float(params_hier[0]),
        'E_pythia': float(params_hier[1]),
        'E_cerebras': float(params_hier[2]),
        'A': float(params_hier[3]),
        'alpha': float(params_hier[4]),
        'B': float(params_hier[5]),
        'beta': float(params_hier[6]),
        'R2': float(r2_hier),
        'RMSE': float(rmse_hier)
    }
elif best_model == 'Ensemble':
    best_params = {
        'model': 'Ensemble',
        'formula': 'Separate Chinchilla per source',
        'source_models': {k: list(map(float, v)) for k, v in source_models.items()},
        'R2': float(r2_ensemble),
        'RMSE': float(rmse_ensemble)
    }
else:
    best_params = {
        'model': best_model,
        'R2': float(best_r2)
    }

with open('best_model_improved.json', 'w') as f:
    json.dump(best_params, f, indent=2)

# Also save comparison table
df_results.to_csv('model_comparison_results.csv', index=False)

print("\n✅ Best model saved: best_model_improved.json")
print("✅ Comparison table saved: model_comparison_results.csv")

print("\n" + "="*70)
print("Experiment Complete!")
print("="*70)
