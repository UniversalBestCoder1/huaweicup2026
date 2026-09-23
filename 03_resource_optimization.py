#!/usr/bin/env python3
"""
阶段3：算力约束下的资源优化
求解最优的 N* 和 D* 配置
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, differential_evolution
import json

sns.set_style("whitegrid")

# ============================================================
# 加载 Scaling Law 参数
# ============================================================

def load_scaling_params():
    """加载已拟合的 Scaling Law 参数"""
    with open('chinchilla_params_final.json', 'r') as f:
        params = json.load(f)

    E = params['E']
    A = params['A']
    alpha = params['alpha']
    B = params['B']
    beta = params['beta']

    print("📊 Scaling Law 参数:")
    print(f"   L(N,D) = {E:.4f} + {A:.4f}/N^{alpha:.4f} + {B:.4f}/D^{beta:.4f}")

    return E, A, alpha, B, beta

def loss_function(N, D, E, A, alpha, B, beta):
    """计算 Loss"""
    return E + A / (N ** alpha) + B / (D ** beta)

# ============================================================
# 优化问题求解
# ============================================================

def optimize_single_budget(C_budget, E, A, alpha, B, beta):
    """
    在给定算力预算 C 下，求解最优的 (N*, D*)

    优化问题:
        min L(N, D) = E + A/N^α + B/D^β
        s.t. 6ND = C
             N > 0, D > 0
    """

    # 方法1: 解析法（Lagrange 乘数）
    # 最优条件: α·A·D/(N^(α+1)) = β·B·N/(D^(β+1))
    # 结合约束 D = C/(6N)

    def objective(log_N):
        """目标函数（对 log(N) 优化以保证 N>0）"""
        N = np.exp(log_N)
        D = C_budget / (6 * N * 1e18)  # 转换单位: N和D是十亿，C是FLOPs

        if D <= 0:
            return 1e10  # 惩罚

        loss = E + A / (N ** alpha) + B / (D ** beta)
        return loss

    # 初始猜测: N ≈ (C/6D_guess)^0.5
    log_N_init = np.log(np.sqrt(C_budget / (6 * 1000 * 1e18)))

    # 优化
    result = minimize(
        objective,
        x0=[log_N_init],
        method='Nelder-Mead',
        options={'maxiter': 10000}
    )

    N_opt = np.exp(result.x[0])
    D_opt = C_budget / (6 * N_opt * 1e18)
    L_opt = loss_function(N_opt, D_opt, E, A, alpha, B, beta)

    return N_opt, D_opt, L_opt

def solve_multiple_budgets(E, A, alpha, B, beta):
    """求解多个算力预算下的最优配置"""

    print("\n" + "="*60)
    print("🔧 求解最优资源配置")
    print("="*60)

    # 算力预算范围 (FLOPs)
    C_budgets = np.logspace(18, 25, 50)  # 10^18 到 10^25 FLOPs

    results = []

    for C in C_budgets:
        N_opt, D_opt, L_opt = optimize_single_budget(C, E, A, alpha, B, beta)

        results.append({
            'C_FLOPs': C,
            'N_params_B': N_opt,
            'D_tokens_B': D_opt,
            'Loss': L_opt,
            'C_ratio_ND': (6 * N_opt * D_opt * 1e18) / C  # 验证约束
        })

    df_results = pd.DataFrame(results)

    # 显示几个典型预算的结果
    print("\n📊 典型算力预算下的最优配置:\n")
    print(f"{'算力预算 (FLOPs)':<25} {'N* (B参数)':<15} {'D* (B tokens)':<18} {'最优Loss':<12}")
    print("-" * 70)

    typical_budgets = [1e19, 1e21, 1e22, 1e23, 1e24, 1e25]
    for C_target in typical_budgets:
        idx = np.argmin(np.abs(df_results['C_FLOPs'] - C_target))
        row = df_results.iloc[idx]
        c_str = f"{row['C_FLOPs']:.2e}".ljust(25)
        print(f"{c_str} {row['N_params_B']:13.2f}  {row['D_tokens_B']:16.2f}  {row['Loss']:10.4f}")

    return df_results

# ============================================================
# 分析最优配置规律
# ============================================================

def analyze_scaling_pattern(df_results):
    """分析最优配置的缩放规律"""

    print("\n" + "="*60)
    print("📈 分析最优配置的缩放规律")
    print("="*60)

    # 计算 N* 和 D* 相对于 C 的缩放指数
    log_C = np.log10(df_results['C_FLOPs'].values)
    log_N = np.log10(df_results['N_params_B'].values)
    log_D = np.log10(df_results['D_tokens_B'].values)

    # 线性拟合 log(N*) ~ log(C) 和 log(D*) ~ log(C)
    from scipy.stats import linregress

    slope_N, intercept_N, r_N, _, _ = linregress(log_C, log_N)
    slope_D, intercept_D, r_D, _, _ = linregress(log_C, log_D)

    print(f"\n缩放规律:")
    print(f"   N* ∝ C^{slope_N:.4f}  (R² = {r_N**2:.4f})")
    print(f"   D* ∝ C^{slope_D:.4f}  (R² = {r_D**2:.4f})")
    print(f"\n理论预期 (Chinchilla):")
    print(f"   N* ∝ C^0.5, D* ∝ C^0.5")
    print(f"\n实际结果:")
    print(f"   N*/D* 比例随 C 的变化: {(slope_N - slope_D):.4f}")

    # 计算 N*/D* 比例
    df_results['N_D_ratio'] = df_results['N_params_B'] / df_results['D_tokens_B']

    return slope_N, slope_D

# ============================================================
# 可视化最优配置
# ============================================================

def visualize_optimal_allocation(df_results, E, A, alpha, B, beta):
    """可视化最优资源配置"""

    fig = plt.figure(figsize=(18, 12))

    # 1. N* 和 D* vs C
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(df_results['C_FLOPs'], df_results['N_params_B'],
             'b-', linewidth=2, label='N* (Model Size)')
    ax1.plot(df_results['C_FLOPs'], df_results['D_tokens_B'],
             'r-', linewidth=2, label='D* (Data Size)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Compute Budget C (FLOPs)', fontsize=12)
    ax1.set_ylabel('Optimal Allocation (Billions)', fontsize=12)
    ax1.set_title('Optimal N* and D* vs Compute Budget', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. N*/D* 比例
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(df_results['C_FLOPs'], df_results['N_D_ratio'], 'g-', linewidth=2)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Compute Budget C (FLOPs)', fontsize=12)
    ax2.set_ylabel('N*/D* Ratio', fontsize=12)
    ax2.set_title('Model-to-Data Ratio', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. 最优 Loss vs C
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(df_results['C_FLOPs'], df_results['Loss'], 'purple', linewidth=2)
    ax3.set_xscale('log')
    ax3.set_xlabel('Compute Budget C (FLOPs)', fontsize=12)
    ax3.set_ylabel('Optimal Loss', fontsize=12)
    ax3.set_title('Best Achievable Loss vs Compute', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. 等高线图: Loss(N, D) 和最优路径
    ax4 = plt.subplot(2, 3, 4)

    N_range = np.logspace(-1, 2, 50)
    D_range = np.logspace(0, 4, 50)
    N_grid, D_grid = np.meshgrid(N_range, D_range)
    L_grid = E + A / (N_grid ** alpha) + B / (D_grid ** beta)

    contour = ax4.contourf(N_grid, D_grid, L_grid, levels=20, cmap='viridis_r', alpha=0.7)
    ax4.contour(N_grid, D_grid, L_grid, levels=10, colors='white', alpha=0.3, linewidths=0.5)

    # 叠加最优路径
    ax4.plot(df_results['N_params_B'], df_results['D_tokens_B'],
             'r-', linewidth=3, label='Optimal Path')
    ax4.scatter(df_results['N_params_B'], df_results['D_tokens_B'],
                c='red', s=30, zorder=5, alpha=0.6)

    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('Model Parameters N (Billions)', fontsize=12)
    ax4.set_ylabel('Training Tokens D (Billions)', fontsize=12)
    ax4.set_title('Loss Landscape & Optimal Frontier', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    plt.colorbar(contour, ax=ax4, label='Loss')

    # 5. 等算力线
    ax5 = plt.subplot(2, 3, 5)

    C_iso_lines = [1e20, 1e21, 1e22, 1e23, 1e24]
    for C in C_iso_lines:
        N_iso = np.logspace(-1, 2, 100)
        D_iso = C / (6 * N_iso * 1e18)
        ax5.plot(N_iso, D_iso, label=f'C={C:.0e}', linewidth=2)

    # 叠加最优点
    ax5.scatter(df_results['N_params_B'], df_results['D_tokens_B'],
                c='red', s=50, zorder=5, label='Optimal Points', alpha=0.7)

    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.set_xlabel('Model Parameters N (Billions)', fontsize=12)
    ax5.set_ylabel('Training Tokens D (Billions)', fontsize=12)
    ax5.set_title('Iso-Compute Lines', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. 边际效用分析
    ax6 = plt.subplot(2, 3, 6)

    # 计算边际效用: dL/dN 和 dL/dD (在最优点)
    N_opt = df_results['N_params_B'].values
    D_opt = df_results['D_tokens_B'].values

    dL_dN = -alpha * A / (N_opt ** (alpha + 1))
    dL_dD = -beta * B / (D_opt ** (beta + 1))

    ax6.plot(df_results['C_FLOPs'], np.abs(dL_dN),
             'b-', linewidth=2, label='|dL/dN| (Model)')
    ax6.plot(df_results['C_FLOPs'], np.abs(dL_dD),
             'r-', linewidth=2, label='|dL/dD| (Data)')

    ax6.set_xscale('log')
    ax6.set_yscale('log')
    ax6.set_xlabel('Compute Budget C (FLOPs)', fontsize=12)
    ax6.set_ylabel('|Marginal Effect|', fontsize=12)
    ax6.set_title('Marginal Utility (should be equal at optimum)',
                  fontsize=14, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output_03_optimal_allocation.png', dpi=300, bbox_inches='tight')
    print("\n✅ 可视化已保存到: output_03_optimal_allocation.png")
    plt.close()

# ============================================================
# 主函数
# ============================================================

def main():
    print("\n🚀 开始资源优化分析...\n")

    # 1. 加载 Scaling Law 参数
    E, A, alpha, B, beta = load_scaling_params()

    # 2. 求解多个预算下的最优配置
    df_results = solve_multiple_budgets(E, A, alpha, B, beta)

    # 3. 分析缩放规律
    slope_N, slope_D = analyze_scaling_pattern(df_results)

    # 4. 可视化
    visualize_optimal_allocation(df_results, E, A, alpha, B, beta)

    # 5. 保存结果
    df_results.to_csv('optimal_allocation_results.csv', index=False)
    print("\n✅ 结果已保存到: optimal_allocation_results.csv")

    print("\n" + "="*60)
    print("✅ 阶段3完成: 资源优化")
    print("="*60)

    print("\n💡 关键发现:")
    print(f"   1. N* 随算力的缩放指数: {slope_N:.3f}")
    print(f"   2. D* 随算力的缩放指数: {slope_D:.3f}")
    print(f"   3. 在最优点，模型和数据的边际效用相等")
    print(f"   4. 实际配置可能偏离理论值 (α={alpha:.3f}, β={beta:.3f})")

if __name__ == "__main__":
    main()
