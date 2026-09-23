#!/usr/bin/env python3
"""
残差分析与模型诊断
分析残差分布的非正态性，并尝试改进
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit, minimize, differential_evolution
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

# 加载数据和参数
import json
with open('chinchilla_params_final.json', 'r') as f:
    params = json.load(f)

E, A, alpha, B, beta = params['E'], params['A'], params['alpha'], params['B'], params['beta']

# 加载数据
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

# 计算预测值和残差
def chinchilla_loss(N, D, E, A, alpha, B, beta):
    return E + A / (N ** alpha) + B / (D ** beta)

L_pred = chinchilla_loss(N, D, E, A, alpha, B, beta)
residuals = L - L_pred

print("="*70)
print("残差分析与诊断")
print("="*70)

# 1. 残差统计
print("\n1. 残差基本统计:")
print(f"   Mean: {residuals.mean():.6f}")
print(f"   Std: {residuals.std():.4f}")
print(f"   Median: {np.median(residuals):.4f}")
print(f"   Skewness: {stats.skew(residuals):.4f}")
print(f"   Kurtosis: {stats.kurtosis(residuals):.4f}")

# 2. 正态性检验
shapiro_stat, shapiro_p = stats.shapiro(residuals)
ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(residuals.mean(), residuals.std()))

print("\n2. 正态性检验:")
print(f"   Shapiro-Wilk test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.6f}")
print(f"   Kolmogorov-Smirnov test: statistic={ks_stat:.4f}, p-value={ks_p:.6f}")
if shapiro_p < 0.05:
    print("   ⚠️  残差显著偏离正态分布 (p < 0.05)")
else:
    print("   ✓  残差接近正态分布")

# 3. 异常值检测
Q1 = np.percentile(residuals, 25)
Q3 = np.percentile(residuals, 75)
IQR = Q3 - Q1
outliers = (residuals < Q1 - 1.5*IQR) | (residuals > Q3 + 1.5*IQR)
n_outliers = outliers.sum()

print(f"\n3. 异常值分析:")
print(f"   Outliers (IQR method): {n_outliers} / {len(residuals)} ({n_outliers/len(residuals)*100:.1f}%)")
if n_outliers > 0:
    print(f"   Max positive residual: {residuals.max():.4f}")
    print(f"   Max negative residual: {residuals.min():.4f}")

# 4. 按数据源分析
print(f"\n4. 按数据源分析残差:")
for source in ['baseline', 'pythia', 'cerebras']:
    mask = all_data['data_source'] == source
    res_source = residuals[mask]
    print(f"   {source.capitalize()}: mean={res_source.mean():.4f}, std={res_source.std():.4f}, n={mask.sum()}")

# 5. 残差与预测变量的关系
corr_N = np.corrcoef(np.log10(N), residuals)[0, 1]
corr_D = np.corrcoef(np.log10(D), residuals)[0, 1]
corr_pred = np.corrcoef(L_pred, residuals)[0, 1]

print(f"\n5. 残差相关性:")
print(f"   Corr(log10(N), residuals) = {corr_N:.4f}")
print(f"   Corr(log10(D), residuals) = {corr_D:.4f}")
print(f"   Corr(L_pred, residuals) = {corr_pred:.4f}")

if abs(corr_N) > 0.3 or abs(corr_D) > 0.3:
    print("   ⚠️  残差与预测变量存在较强相关性，提示模型存在系统性偏差")

# 6. 可视化诊断
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 6.1 残差直方图 + 正态分布拟合
ax = axes[0, 0]
n, bins, patches = ax.hist(residuals, bins=30, density=True, alpha=0.7,
                           edgecolor='black', color='steelblue')
mu, sigma = residuals.mean(), residuals.std()
x = np.linspace(residuals.min(), residuals.max(), 100)
ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')
ax.axvline(0, color='k', linestyle='--', linewidth=2)
ax.set_xlabel('Residual', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title(f'Residual Distribution\nSkewness={stats.skew(residuals):.2f}, Kurtosis={stats.kurtosis(residuals):.2f}',
             fontsize=11, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 6.2 Q-Q plot
ax = axes[0, 1]
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title('Q-Q Plot (Normal)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

# 6.3 残差 vs 拟合值
ax = axes[0, 2]
colors_map = {'baseline': 'red', 'pythia': 'blue', 'cerebras': 'green'}
for source, color in colors_map.items():
    mask = all_data['data_source'] == source
    ax.scatter(L_pred[mask], residuals[mask], c=color, alpha=0.6, s=30, label=source.capitalize())
ax.axhline(0, color='k', linestyle='--', linewidth=2)
ax.axhline(1.5*residuals.std(), color='r', linestyle=':', linewidth=1, alpha=0.5)
ax.axhline(-1.5*residuals.std(), color='r', linestyle=':', linewidth=1, alpha=0.5)
ax.set_xlabel('Fitted Loss', fontsize=11)
ax.set_ylabel('Residual', fontsize=11)
ax.set_title('Residual vs Fitted Values', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 6.4 残差 vs log(N)
ax = axes[1, 0]
for source, color in colors_map.items():
    mask = all_data['data_source'] == source
    ax.scatter(np.log10(N[mask]), residuals[mask], c=color, alpha=0.6, s=30, label=source.capitalize())
ax.axhline(0, color='k', linestyle='--', linewidth=2)
# 添加局部回归线（LOWESS）
from scipy.signal import savgol_filter
sort_idx = np.argsort(np.log10(N))
if len(residuals) > 10:
    smoothed = savgol_filter(residuals[sort_idx], min(51, len(residuals)//2*2-1), 3)
    ax.plot(np.log10(N)[sort_idx], smoothed, 'orange', linewidth=2, label='Trend')
ax.set_xlabel('log10(N)', fontsize=11)
ax.set_ylabel('Residual', fontsize=11)
ax.set_title(f'Residual vs Model Size\nCorr={corr_N:.3f}', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 6.5 残差 vs log(D)
ax = axes[1, 1]
for source, color in colors_map.items():
    mask = all_data['data_source'] == source
    ax.scatter(np.log10(D[mask]), residuals[mask], c=color, alpha=0.6, s=30, label=source.capitalize())
ax.axhline(0, color='k', linestyle='--', linewidth=2)
sort_idx = np.argsort(np.log10(D))
if len(residuals) > 10:
    smoothed = savgol_filter(residuals[sort_idx], min(51, len(residuals)//2*2-1), 3)
    ax.plot(np.log10(D)[sort_idx], smoothed, 'orange', linewidth=2, label='Trend')
ax.set_xlabel('log10(D)', fontsize=11)
ax.set_ylabel('Residual', fontsize=11)
ax.set_title(f'Residual vs Data Size\nCorr={corr_D:.3f}', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 6.6 残差箱线图（按数据源）
ax = axes[1, 2]
residual_by_source = [residuals[all_data['data_source'] == s] for s in ['baseline', 'pythia', 'cerebras']]
bp = ax.boxplot(residual_by_source, labels=['Baseline', 'Pythia', 'Cerebras'],
                patch_artist=True)
for patch, color in zip(bp['boxes'], ['red', 'blue', 'green']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.axhline(0, color='k', linestyle='--', linewidth=2)
ax.set_ylabel('Residual', fontsize=11)
ax.set_title('Residual Distribution by Source', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('residual_analysis_diagnostic.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Diagnostic plot saved: residual_analysis_diagnostic.png")

# 7. 诊断结论
print("\n" + "="*70)
print("诊断结论与建议")
print("="*70)

issues = []
if abs(stats.skew(residuals)) > 0.5:
    issues.append("残差分布偏斜（skewness > 0.5）")
if abs(stats.kurtosis(residuals)) > 1:
    issues.append("残差分布尾部过厚或过薄（|kurtosis| > 1）")
if shapiro_p < 0.05:
    issues.append("残差显著偏离正态分布（Shapiro-Wilk p < 0.05）")
if abs(corr_N) > 0.3 or abs(corr_D) > 0.3:
    issues.append("残差与预测变量存在较强相关性")
if n_outliers / len(residuals) > 0.1:
    issues.append(f"异常值比例较高（{n_outliers/len(residuals)*100:.1f}%）")

if issues:
    print("\n⚠️  发现的问题:")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")

    print("\n💡 可能的原因:")
    print("   1. 数据来源差异: baseline, pythia, cerebras 可能有系统性差异")
    print("   2. 模型形式: Chinchilla 模型可能不完全适合所有数据")
    print("   3. 数据质量: 部分数据点可能未完全收敛或有测量误差")
    print("   4. 缺少交互项: 可能需要 N-D 交互项")

    print("\n🔧 改进建议:")
    print("   1. 考虑分层建模（按数据源分别拟合）")
    print("   2. 尝试稳健回归（Huber loss）降低异常值影响")
    print("   3. 添加交互项: L = E + A/N^α + B/D^β + C/(ND)^γ")
    print("   4. 使用更灵活的模型（如 GPR, ensemble）")
    print("   5. 数据清洗：移除明显的异常值后重新拟合")
else:
    print("\n✅ 残差分布合理，模型拟合质量良好")

print("\n注意: 对于数学建模竞赛，R²=0.51 已经是不错的结果")
print("      残差的非正态性可能反映了真实数据的复杂性")
print("      建议在报告中说明诊断结果并讨论模型局限性")
