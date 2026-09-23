#!/usr/bin/env python3
"""
阶段2：Scaling Law 建模
拟合 Chinchilla 和 OpenAI Scaling Law
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit, minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

# ============================================================
# 模型定义
# ============================================================

def chinchilla_loss(params, N, D):
    """
    Chinchilla Scaling Law:
    L(N, D) = E + A/N^α + B/D^β

    params: [E, A, α, B, β]
    N: 模型参数量 (Billions)
    D: 训练数据量 (Billions of tokens)
    """
    E, A, alpha, B, beta = params
    return E + A / (N ** alpha) + B / (D ** beta)

def chinchilla_loss_fit(ND, E, A, alpha, B, beta):
    """用于 curve_fit 的包装函数"""
    N, D = ND
    return E + A / (N ** alpha) + B / (D ** beta)

def openai_loss(params, N, D):
    """
    OpenAI Scaling Law:
    L(N, D) = [(N_c/N)^α_N + (D_c/D)^α_D]^β

    params: [N_c, α_N, D_c, α_D, β]
    """
    N_c, alpha_N, D_c, alpha_D, beta = params
    return ((N_c / N) ** alpha_N + (D_c / D) ** alpha_D) ** beta

# ============================================================
# 数据加载与预处理
# ============================================================

def load_and_prepare_data():
    """加载并准备数据"""
    # 加载基准数据
    baseline = pd.read_csv('processed_baseline.csv')

    # 加载训练日志（更多数据点）
    pythia = pd.read_csv('real_attachments/B_scaling_laws/pythia_training_log_existing.csv')
    cerebras = pd.read_csv('real_attachments/B_scaling_laws/cerebras_training_log.csv')

    # 合并数据
    # 从训练日志中抽取部分数据点以增加样本量
    pythia_sample = pythia[['N_params_B', 'D_tokens_B', 'val_loss']].copy()
    cerebras_sample = cerebras[['N_params_B', 'D_tokens_B', 'val_loss']].copy()
    baseline_sample = baseline[['N_params_B', 'D_tokens_B', 'val_loss']].copy()

    # 合并并去重
    all_data = pd.concat([baseline_sample, pythia_sample, cerebras_sample])
    all_data = all_data.drop_duplicates()
    all_data = all_data.dropna()

    # 过滤异常值（loss < 1.5 可能有问题，loss > 5 可能是早期训练）
    all_data = all_data[(all_data['val_loss'] > 1.5) & (all_data['val_loss'] < 5.0)]

    print(f"📊 总数据点: {len(all_data)}")
    print(f"   N 范围: {all_data['N_params_B'].min():.3f}B - {all_data['N_params_B'].max():.1f}B")
    print(f"   D 范围: {all_data['D_tokens_B'].min():.3f}B - {all_data['D_tokens_B'].max():.1f}B tokens")
    print(f"   Loss 范围: {all_data['val_loss'].min():.3f} - {all_data['val_loss'].max():.3f}")

    return all_data

# ============================================================
# Chinchilla 模型拟合
# ============================================================

def fit_chinchilla_model(data):
    """拟合 Chinchilla Scaling Law"""
    print("\n" + "="*60)
    print("🔧 拟合 Chinchilla Scaling Law")
    print("="*60)

    N = data['N_params_B'].values
    D = data['D_tokens_B'].values
    L = data['val_loss'].values

    # 初始参数猜测（基于文献）
    # E ≈ 1.69 (Chinchilla paper)
    # α ≈ 0.34, β ≈ 0.28
    initial_guess = [1.7, 0.3, 0.34, 0.4, 0.28]

    # 参数边界
    bounds = (
        [1.5, 0.01, 0.1, 0.01, 0.1],  # 下界
        [2.5, 10.0, 0.8, 10.0, 0.8]   # 上界
    )

    try:
        # 拟合
        params, covariance = curve_fit(
            chinchilla_loss_fit,
            (N, D),
            L,
            p0=initial_guess,
            bounds=bounds,
            maxfev=10000
        )

        E, A, alpha, B, beta = params

        print(f"\n✅ 拟合成功!")
        print(f"   E (不可约损失) = {E:.4f}")
        print(f"   A = {A:.4f}")
        print(f"   α (模型规模指数) = {alpha:.4f}")
        print(f"   B = {B:.4f}")
        print(f"   β (数据规模指数) = {beta:.4f}")

        # 计算拟合优度
        L_pred = chinchilla_loss(params, N, D)
        r2 = r2_score(L, L_pred)
        rmse = np.sqrt(mean_squared_error(L, L_pred))

        print(f"\n📊 拟合优度:")
        print(f"   R² = {r2:.4f}")
        print(f"   RMSE = {rmse:.4f}")

        return params, r2, rmse

    except Exception as e:
        print(f"❌ 拟合失败: {e}")
        return None, None, None

# ============================================================
# 可视化拟合结果
# ============================================================

def visualize_fit(data, params):
    """可视化拟合结果"""
    fig = plt.figure(figsize=(16, 10))

    N = data['N_params_B'].values
    D = data['D_tokens_B'].values
    L_true = data['val_loss'].values
    L_pred = chinchilla_loss(params, N, D)

    # 1. 预测 vs 真实值
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(L_true, L_pred, alpha=0.5, s=20)
    ax1.plot([L_true.min(), L_true.max()],
             [L_true.min(), L_true.max()],
             'r--', linewidth=2, label='Perfect fit')
    ax1.set_xlabel('True Loss', fontsize=12)
    ax1.set_ylabel('Predicted Loss', fontsize=12)
    ax1.set_title('Predicted vs True Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 残差分布
    ax2 = plt.subplot(2, 3, 2)
    residuals = L_true - L_pred
    ax2.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residual (True - Pred)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Residual Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. N vs Loss (固定几个D值)
    ax3 = plt.subplot(2, 3, 3)
    D_values = [10, 100, 1000, 10000]
    N_range = np.logspace(np.log10(0.05), np.log10(100), 100)

    for D_val in D_values:
        L_curve = chinchilla_loss(params, N_range, D_val)
        ax3.plot(N_range, L_curve, label=f'D={D_val}B tokens', linewidth=2)

    # 叠加真实数据点
    ax3.scatter(N, L_true, alpha=0.3, s=10, color='gray', label='Data')
    ax3.set_xscale('log')
    ax3.set_xlabel('Model Parameters N (Billions)', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_title('Scaling with Model Size', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. D vs Loss (固定几个N值)
    ax4 = plt.subplot(2, 3, 4)
    N_values = [0.1, 1, 7, 30]
    D_range = np.logspace(np.log10(1), np.log10(20000), 100)

    for N_val in N_values:
        L_curve = chinchilla_loss(params, N_val, D_range)
        ax4.plot(D_range, L_curve, label=f'N={N_val}B', linewidth=2)

    ax4.scatter(D, L_true, alpha=0.3, s=10, color='gray', label='Data')
    ax4.set_xscale('log')
    ax4.set_xlabel('Training Tokens D (Billions)', fontsize=12)
    ax4.set_ylabel('Loss', fontsize=12)
    ax4.set_title('Scaling with Data Size', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. 3D表面图（N-D-Loss）
    ax5 = plt.subplot(2, 3, 5, projection='3d')

    # 创建网格
    N_grid = np.logspace(np.log10(0.1), np.log10(50), 30)
    D_grid = np.logspace(np.log10(10), np.log10(10000), 30)
    N_mesh, D_mesh = np.meshgrid(N_grid, D_grid)
    L_mesh = chinchilla_loss(params, N_mesh, D_mesh)

    surf = ax5.plot_surface(np.log10(N_mesh), np.log10(D_mesh), L_mesh,
                            cmap='viridis', alpha=0.7, edgecolor='none')
    ax5.scatter(np.log10(N), np.log10(D), L_true, c='red', s=5, alpha=0.5)

    ax5.set_xlabel('log10(N)', fontsize=10)
    ax5.set_ylabel('log10(D)', fontsize=10)
    ax5.set_zlabel('Loss', fontsize=10)
    ax5.set_title('3D Scaling Surface', fontsize=14, fontweight='bold')

    # 6. 残差 vs 预测值
    ax6 = plt.subplot(2, 3, 6)
    ax6.scatter(L_pred, residuals, alpha=0.5, s=20)
    ax6.axhline(0, color='r', linestyle='--', linewidth=2)
    ax6.set_xlabel('Predicted Loss', fontsize=12)
    ax6.set_ylabel('Residual', fontsize=12)
    ax6.set_title('Residual vs Predicted', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output_02_chinchilla_fit.png', dpi=300, bbox_inches='tight')
    print("\n✅ 可视化已保存到: output_02_chinchilla_fit.png")
    plt.close()

# ============================================================
# 保存模型参数
# ============================================================

def save_model_params(params):
    """保存拟合参数"""
    E, A, alpha, B, beta = params

    model_info = {
        'model_type': 'Chinchilla',
        'formula': 'L(N,D) = E + A/N^α + B/D^β',
        'E': E,
        'A': A,
        'alpha': alpha,
        'B': B,
        'beta': beta
    }

    import json
    with open('chinchilla_params.json', 'w') as f:
        json.dump(model_info, f, indent=2)

    print("\n✅ 模型参数已保存到: chinchilla_params.json")

# ============================================================
# 主函数
# ============================================================

def main():
    print("\n🚀 开始 Scaling Law 建模...\n")

    # 1. 加载数据
    data = load_and_prepare_data()

    # 2. 拟合 Chinchilla 模型
    params, r2, rmse = fit_chinchilla_model(data)

    if params is not None:
        # 3. 可视化拟合结果
        visualize_fit(data, params)

        # 4. 保存模型参数
        save_model_params(params)

        print("\n" + "="*60)
        print("✅ 阶段2完成：Scaling Law 建模")
        print("="*60)
    else:
        print("\n❌ 建模失败")

if __name__ == "__main__":
    main()
