#!/usr/bin/env python3
"""
阶段2优化：改进 Scaling Law 拟合
使用收敛点数据 + 更好的拟合策略
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit, minimize, differential_evolution
from sklearn.metrics import r2_score, mean_squared_error
import json
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

# ============================================================
# 数据准备：使用高质量收敛点
# ============================================================

def load_high_quality_data():
    """加载高质量数据（收敛点 + 训练轨迹末期）"""
    print("📊 加载高质量数据...")

    # 1. 基准收敛点（最高质量）
    baseline = pd.read_csv('processed_baseline.csv')
    baseline['data_source'] = 'baseline_converged'

    # 2. Pythia 训练轨迹的收敛点
    pythia = pd.read_csv('real_attachments/B_scaling_laws/pythia_training_log_existing.csv')
    # 取每个模型的最后几个checkpoint
    pythia_converged = pythia.groupby('N_params_B').tail(5)[['N_params_B', 'D_tokens_B', 'val_loss']]
    pythia_converged['data_source'] = 'pythia_late'

    # 3. Cerebras 训练轨迹的收敛点
    cerebras = pd.read_csv('real_attachments/B_scaling_laws/cerebras_training_log.csv')
    cerebras_converged = cerebras.groupby('N_params_B').tail(5)[['N_params_B', 'D_tokens_B', 'val_loss']]
    cerebras_converged['data_source'] = 'cerebras_late'

    # 合并
    all_data = pd.concat([baseline, pythia_converged, cerebras_converged])

    # 数据清洗
    all_data = all_data.dropna(subset=['N_params_B', 'D_tokens_B', 'val_loss'])
    all_data = all_data[(all_data['val_loss'] > 1.5) & (all_data['val_loss'] < 5.0)]
    all_data = all_data[(all_data['N_params_B'] > 0) & (all_data['D_tokens_B'] > 0)]

    print(f"   总数据点: {len(all_data)}")
    print(f"   - Baseline: {len(all_data[all_data['data_source']=='baseline_converged'])}")
    print(f"   - Pythia: {len(all_data[all_data['data_source']=='pythia_late'])}")
    print(f"   - Cerebras: {len(all_data[all_data['data_source']=='cerebras_late'])}")

    return all_data

# ============================================================
# 改进的拟合方法
# ============================================================

def chinchilla_loss_vectorized(ND, E, A, alpha, B, beta):
    """Chinchilla模型（向量化）"""
    N, D = ND
    return E + A / (N ** alpha) + B / (D ** beta)

def fit_chinchilla_improved(data, use_weights=True):
    """改进的拟合方法"""
    print("\n" + "="*60)
    print("🔧 改进的 Chinchilla Scaling Law 拟合")
    print("="*60)

    N = data['N_params_B'].values
    D = data['D_tokens_B'].values
    L = data['val_loss'].values

    # 设置权重：baseline数据权重更高
    if use_weights:
        weights = np.ones(len(data))
        weights[data['data_source'] == 'baseline_converged'] = 3.0
        weights = weights / weights.sum() * len(weights)  # 归一化
        print(f"   使用加权拟合（baseline权重=3.0）")
    else:
        weights = None

    # 方法1：使用 curve_fit
    print("\n[方法1] scipy.optimize.curve_fit")
    try:
        initial_guess = [1.7, 0.5, 0.3, 0.3, 0.3]
        bounds = (
            [1.5, 0.01, 0.05, 0.01, 0.05],
            [2.5, 20.0, 1.0, 20.0, 1.0]
        )

        params_cf, cov = curve_fit(
            chinchilla_loss_vectorized,
            (N, D),
            L,
            p0=initial_guess,
            bounds=bounds,
            sigma=1.0/weights if use_weights else None,
            maxfev=50000
        )

        L_pred_cf = chinchilla_loss_vectorized((N, D), *params_cf)
        r2_cf = r2_score(L, L_pred_cf)
        rmse_cf = np.sqrt(mean_squared_error(L, L_pred_cf))

        print(f"   R² = {r2_cf:.4f}, RMSE = {rmse_cf:.4f}")
        print(f"   参数: E={params_cf[0]:.4f}, A={params_cf[1]:.4f}, α={params_cf[2]:.4f}, "
              f"B={params_cf[3]:.4f}, β={params_cf[4]:.4f}")

    except Exception as e:
        print(f"   ❌ 失败: {e}")
        params_cf, r2_cf, rmse_cf = None, -999, 999

    # 方法2：使用差分进化算法（全局优化）
    print("\n[方法2] scipy.optimize.differential_evolution")

    def objective_de(params):
        E, A, alpha, B, beta = params
        L_pred = E + A / (N ** alpha) + B / (D ** beta)
        if use_weights:
            return np.sum(weights * (L - L_pred) ** 2)
        else:
            return np.sum((L - L_pred) ** 2)

    try:
        bounds_de = [
            (1.5, 2.5),   # E
            (0.01, 20),   # A
            (0.05, 1.0),  # alpha
            (0.01, 20),   # B
            (0.05, 1.0)   # beta
        ]

        result_de = differential_evolution(
            objective_de,
            bounds_de,
            maxiter=1000,
            popsize=30,
            seed=42,
            polish=True
        )

        params_de = result_de.x
        L_pred_de = chinchilla_loss_vectorized((N, D), *params_de)
        r2_de = r2_score(L, L_pred_de)
        rmse_de = np.sqrt(mean_squared_error(L, L_pred_de))

        print(f"   R² = {r2_de:.4f}, RMSE = {rmse_de:.4f}")
        print(f"   参数: E={params_de[0]:.4f}, A={params_de[1]:.4f}, α={params_de[2]:.4f}, "
              f"B={params_de[3]:.4f}, β={params_de[4]:.4f}")

    except Exception as e:
        print(f"   ❌ 失败: {e}")
        params_de, r2_de, rmse_de = None, -999, 999

    # 选择最佳模型
    print("\n" + "="*60)
    if r2_cf > r2_de:
        print(f"✅ 选择方法1（curve_fit）: R²={r2_cf:.4f}")
        best_params = params_cf
        best_r2 = r2_cf
        best_rmse = rmse_cf
    else:
        print(f"✅ 选择方法2（differential_evolution）: R²={r2_de:.4f}")
        best_params = params_de
        best_r2 = r2_de
        best_rmse = rmse_de

    E, A, alpha, B, beta = best_params
    print(f"\n📊 最终参数:")
    print(f"   E (不可约损失) = {E:.4f}")
    print(f"   A = {A:.4f}")
    print(f"   α (模型规模指数) = {alpha:.4f}")
    print(f"   B = {B:.4f}")
    print(f"   β (数据规模指数) = {beta:.4f}")
    print(f"\n   R² = {best_r2:.4f}")
    print(f"   RMSE = {best_rmse:.4f}")

    return best_params, best_r2, best_rmse

# ============================================================
# 可视化（增强版）
# ============================================================

def visualize_fit_enhanced(data, params):
    """增强版可视化"""
    fig = plt.figure(figsize=(18, 12))

    N = data['N_params_B'].values
    D = data['D_tokens_B'].values
    L_true = data['val_loss'].values
    L_pred = chinchilla_loss_vectorized((N, D), *params)

    # 按数据源着色
    colors = {'baseline_converged': 'red', 'pythia_late': 'blue', 'cerebras_late': 'green'}
    data['color'] = data['data_source'].map(colors)

    # 1. 预测 vs 真实值（按数据源着色）
    ax1 = plt.subplot(2, 3, 1)
    for source, color in colors.items():
        mask = data['data_source'] == source
        ax1.scatter(L_true[mask], L_pred[mask], c=color, label=source, alpha=0.6, s=30)
    ax1.plot([L_true.min(), L_true.max()],
             [L_true.min(), L_true.max()],
             'k--', linewidth=2, label='Perfect fit')
    ax1.set_xlabel('True Loss', fontsize=12)
    ax1.set_ylabel('Predicted Loss', fontsize=12)
    ax1.set_title('Predicted vs True Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. 残差分析
    ax2 = plt.subplot(2, 3, 2)
    residuals = L_true - L_pred
    ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    ax2.axvline(0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residual (True - Pred)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Residual Distribution\nMean={residuals.mean():.4f}, Std={residuals.std():.4f}',
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. N vs Loss（多条D曲线）
    ax3 = plt.subplot(2, 3, 3)
    D_values = [10, 50, 200, 1000, 5000]
    N_range = np.logspace(-1, 2, 100)

    for D_val in D_values:
        L_curve = chinchilla_loss_vectorized((N_range, D_val), *params)
        ax3.plot(N_range, L_curve, label=f'D={D_val}B', linewidth=2)

    # 叠加数据点
    for source, color in colors.items():
        mask = data['data_source'] == source
        ax3.scatter(N[mask], L_true[mask], c=color, alpha=0.4, s=15)

    ax3.set_xscale('log')
    ax3.set_xlabel('Model Parameters N (Billions)', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_title('Scaling with Model Size', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. D vs Loss（多条N曲线）
    ax4 = plt.subplot(2, 3, 4)
    N_values = [0.1, 0.5, 1, 7, 30]
    D_range = np.logspace(0, 4, 100)

    for N_val in N_values:
        L_curve = chinchilla_loss_vectorized((N_val, D_range), *params)
        ax4.plot(D_range, L_curve, label=f'N={N_val}B', linewidth=2)

    for source, color in colors.items():
        mask = data['data_source'] == source
        ax4.scatter(D[mask], L_true[mask], c=color, alpha=0.4, s=15)

    ax4.set_xscale('log')
    ax4.set_xlabel('Training Tokens D (Billions)', fontsize=12)
    ax4.set_ylabel('Loss', fontsize=12)
    ax4.set_title('Scaling with Data Size', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. 残差 vs N（检查偏差）
    ax5 = plt.subplot(2, 3, 5)
    for source, color in colors.items():
        mask = data['data_source'] == source
        ax5.scatter(N[mask], residuals[mask], c=color, alpha=0.6, s=20, label=source)
    ax5.axhline(0, color='k', linestyle='--', linewidth=2)
    ax5.set_xscale('log')
    ax5.set_xlabel('Model Parameters N (Billions)', fontsize=12)
    ax5.set_ylabel('Residual', fontsize=12)
    ax5.set_title('Residual vs Model Size', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. 残差 vs D（检查偏差）
    ax6 = plt.subplot(2, 3, 6)
    for source, color in colors.items():
        mask = data['data_source'] == source
        ax6.scatter(D[mask], residuals[mask], c=color, alpha=0.6, s=20, label=source)
    ax6.axhline(0, color='k', linestyle='--', linewidth=2)
    ax6.set_xscale('log')
    ax6.set_xlabel('Training Tokens D (Billions)', fontsize=12)
    ax6.set_ylabel('Residual', fontsize=12)
    ax6.set_title('Residual vs Data Size', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output_02b_chinchilla_fit_improved.png', dpi=300, bbox_inches='tight')
    print("\n✅ 可视化已保存到: output_02b_chinchilla_fit_improved.png")
    plt.close()

# ============================================================
# 主函数
# ============================================================

def main():
    print("\n🚀 开始改进的 Scaling Law 建模...\n")

    # 1. 加载高质量数据
    data = load_high_quality_data()

    # 2. 改进的拟合
    params, r2, rmse = fit_chinchilla_improved(data, use_weights=True)

    # 3. 可视化
    visualize_fit_enhanced(data, params)

    # 4. 保存最终参数
    model_info = {
        'model_type': 'Chinchilla_Improved',
        'formula': 'L(N,D) = E + A/N^α + B/D^β',
        'E': float(params[0]),
        'A': float(params[1]),
        'alpha': float(params[2]),
        'B': float(params[3]),
        'beta': float(params[4]),
        'R2': float(r2),
        'RMSE': float(rmse)
    }

    with open('chinchilla_params_final.json', 'w') as f:
        json.dump(model_info, f, indent=2)

    print("\n✅ 最终参数已保存到: chinchilla_params_final.json")
    print("\n" + "="*60)
    print("✅ 阶段2优化完成")
    print("="*60)

if __name__ == "__main__":
    main()
