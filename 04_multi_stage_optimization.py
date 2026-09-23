#!/usr/bin/env python3
"""
阶段4：多阶段训练优化
比较单阶段 vs 多阶段训练策略
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, differential_evolution
import json

sns.set_style("whitegrid")

# ============================================================
# 加载参数
# ============================================================

def load_params():
    """加载 Scaling Law 参数"""
    with open('chinchilla_params_final.json', 'r') as f:
        params = json.load(f)

    E = params['E']
    A = params['A']
    alpha = params['alpha']
    B = params['B']
    beta = params['beta']

    return E, A, alpha, B, beta

def loss_function(N, D, E, A, alpha, B, beta):
    """计算 Loss"""
    return E + A / (N ** alpha) + B / (D ** beta)

# ============================================================
# 多阶段训练策略
# ============================================================

def two_stage_training(C_total, E, A, alpha, B, beta, strategy='continue'):
    """
    两阶段训练策略

    参数:
        C_total: 总算力预算
        strategy: 'continue' (继续训练) 或 'expand' (扩展模型)

    继续训练策略:
        Stage 1: N₁, D₁, 消耗 C₁
        Stage 2: N₂=N₁, D₂=D₁+ΔD, 消耗 C₂ = 6N₁·ΔD
        总约束: C₁ + C₂ = C_total

    扩展模型策略:
        Stage 1: N₁, D₁, 消耗 C₁
        Stage 2: N₂>N₁, D₂, 消耗 C₂
        假设: 部分知识迁移，等效数据量 D_eff = D₁·η + D₂
    """

    if strategy == 'continue':
        # 继续训练：同一模型，增加数据
        def objective(x):
            """x = [log(N), fraction]，fraction 是第一阶段的算力比例"""
            log_N = x[0]
            f1 = x[1]  # 第一阶段算力占比

            if f1 <= 0 or f1 >= 1:
                return 1e10

            N = np.exp(log_N)
            C1 = C_total * f1
            C2 = C_total * (1 - f1)

            # 第一阶段
            D1 = C1 / (6 * N * 1e18)
            if D1 <= 0:
                return 1e10

            # 第二阶段（继续训练）
            delta_D = C2 / (6 * N * 1e18)
            D_total = D1 + delta_D

            # 最终 loss
            loss = loss_function(N, D_total, E, A, alpha, B, beta)
            return loss

        # 优化
        result = minimize(
            objective,
            x0=[np.log(10), 0.5],
            bounds=[(-5, 10), (0.1, 0.9)],
            method='L-BFGS-B'
        )

        log_N_opt, f1_opt = result.x
        N_opt = np.exp(log_N_opt)
        C1_opt = C_total * f1_opt
        C2_opt = C_total * (1 - f1_opt)
        D1_opt = C1_opt / (6 * N_opt * 1e18)
        D2_opt = C2_opt / (6 * N_opt * 1e18)
        D_total = D1_opt + D2_opt
        loss_final = loss_function(N_opt, D_total, E, A, alpha, B, beta)

        return {
            'strategy': 'continue',
            'N1': N_opt,
            'D1': D1_opt,
            'N2': N_opt,
            'D2': D2_opt,
            'D_total': D_total,
            'C1': C1_opt,
            'C2': C2_opt,
            'loss': loss_final
        }

    elif strategy == 'expand':
        # 扩展策略：先训练小模型，再扩展到大模型
        def objective(x):
            """x = [log(N1), log(N2), f1, eta]"""
            log_N1, log_N2, f1, eta = x

            if f1 <= 0.1 or f1 >= 0.9:
                return 1e10
            if eta <= 0 or eta >= 1:
                return 1e10

            N1 = np.exp(log_N1)
            N2 = np.exp(log_N2)

            if N2 <= N1:  # N2 必须 > N1
                return 1e10

            C1 = C_total * f1
            C2 = C_total * (1 - f1)

            D1 = C1 / (6 * N1 * 1e18)
            D2 = C2 / (6 * N2 * 1e18)

            if D1 <= 0 or D2 <= 0:
                return 1e10

            # 等效数据量（知识迁移）
            D_eff = D1 * eta + D2

            # 最终 loss
            loss = loss_function(N2, D_eff, E, A, alpha, B, beta)
            return loss

        # 优化
        result = minimize(
            objective,
            x0=[np.log(1), np.log(10), 0.3, 0.5],
            bounds=[(-2, 8), (0, 10), (0.1, 0.9), (0.1, 0.9)],
            method='L-BFGS-B'
        )

        log_N1, log_N2, f1, eta = result.x
        N1 = np.exp(log_N1)
        N2 = np.exp(log_N2)
        C1 = C_total * f1
        C2 = C_total * (1 - f1)
        D1 = C1 / (6 * N1 * 1e18)
        D2 = C2 / (6 * N2 * 1e18)
        D_eff = D1 * eta + D2
        loss_final = loss_function(N2, D_eff, E, A, alpha, B, beta)

        return {
            'strategy': 'expand',
            'N1': N1,
            'D1': D1,
            'N2': N2,
            'D2': D2,
            'D_eff': D_eff,
            'C1': C1,
            'C2': C2,
            'eta': eta,
            'loss': loss_final
        }

def compare_strategies(C_budgets, E, A, alpha, B, beta):
    """比较不同训练策略"""

    print("\n" + "="*60)
    print("🔧 比较训练策略")
    print("="*60)

    results = []

    for C in C_budgets:
        # 单阶段（baseline）
        N_single = np.exp(minimize(
            lambda log_N: loss_function(
                np.exp(log_N),
                C / (6 * np.exp(log_N) * 1e18),
                E, A, alpha, B, beta
            ),
            x0=[np.log(10)],
            method='Nelder-Mead'
        ).x[0])
        D_single = C / (6 * N_single * 1e18)
        loss_single = loss_function(N_single, D_single, E, A, alpha, B, beta)

        # 两阶段：继续训练
        result_continue = two_stage_training(C, E, A, alpha, B, beta, 'continue')

        # 两阶段：扩展模型
        result_expand = two_stage_training(C, E, A, alpha, B, beta, 'expand')

        results.append({
            'C_FLOPs': C,
            'single_N': N_single,
            'single_D': D_single,
            'single_loss': loss_single,
            'continue_loss': result_continue['loss'],
            'expand_loss': result_expand['loss'],
            'continue_improvement': (loss_single - result_continue['loss']) / loss_single * 100,
            'expand_improvement': (loss_single - result_expand['loss']) / loss_single * 100
        })

    df = pd.DataFrame(results)

    print("\n📊 策略比较结果:\n")
    print(f"{'算力 (FLOPs)':<15} {'单阶段Loss':<15} {'继续训练Loss':<18} {'扩展模型Loss':<18} {'继续改进%':<12} {'扩展改进%'}")
    print("-" * 100)
    for _, row in df.iterrows():
        c_str = f"{row['C_FLOPs']:.1e}"
        print(f"{c_str:<15} {row['single_loss']:>13.4f}  {row['continue_loss']:>16.4f}  "
              f"{row['expand_loss']:>16.4f}  {row['continue_improvement']:>10.2f}  {row['expand_improvement']:>10.2f}")

    return df

# ============================================================
# 可视化
# ============================================================

def visualize_multi_stage(df_comparison):
    """可视化多阶段策略"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Loss 比较
    ax = axes[0, 0]
    ax.plot(df_comparison['C_FLOPs'], df_comparison['single_loss'],
            'b-', linewidth=2, marker='o', label='Single-Stage', markersize=6)
    ax.plot(df_comparison['C_FLOPs'], df_comparison['continue_loss'],
            'g--', linewidth=2, marker='s', label='Two-Stage (Continue)', markersize=6)
    ax.plot(df_comparison['C_FLOPs'], df_comparison['expand_loss'],
            'r-.', linewidth=2, marker='^', label='Two-Stage (Expand)', markersize=6)
    ax.set_xscale('log')
    ax.set_xlabel('Compute Budget (FLOPs)', fontsize=12)
    ax.set_ylabel('Final Loss', fontsize=12)
    ax.set_title('Loss Comparison: Single vs Multi-Stage', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. 改进百分比
    ax = axes[0, 1]
    ax.plot(df_comparison['C_FLOPs'], df_comparison['continue_improvement'],
            'g-', linewidth=2, marker='s', label='Continue Training', markersize=6)
    ax.plot(df_comparison['C_FLOPs'], df_comparison['expand_improvement'],
            'r-', linewidth=2, marker='^', label='Expand Model', markersize=6)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xscale('log')
    ax.set_xlabel('Compute Budget (FLOPs)', fontsize=12)
    ax.set_ylabel('Improvement over Single-Stage (%)', fontsize=12)
    ax.set_title('Multi-Stage Strategy Benefits', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3. N 配置比较
    ax = axes[1, 0]
    ax.plot(df_comparison['C_FLOPs'], df_comparison['single_N'],
            'b-', linewidth=2, marker='o', label='Single-Stage N', markersize=6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Compute Budget (FLOPs)', fontsize=12)
    ax.set_ylabel('Model Size N (Billions)', fontsize=12)
    ax.set_title('Model Size Scaling', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4. D 配置比较
    ax = axes[1, 1]
    ax.plot(df_comparison['C_FLOPs'], df_comparison['single_D'],
            'b-', linewidth=2, marker='o', label='Single-Stage D', markersize=6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Compute Budget (FLOPs)', fontsize=12)
    ax.set_ylabel('Training Tokens D (Billions)', fontsize=12)
    ax.set_title('Data Size Scaling', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output_04_multi_stage_strategy.png', dpi=300, bbox_inches='tight')
    print("\n✅ 可视化已保存到: output_04_multi_stage_strategy.png")
    plt.close()

# ============================================================
# 主函数
# ============================================================

def main():
    print("\n🚀 开始多阶段训练优化分析...\n")

    # 1. 加载参数
    E, A, alpha, B, beta = load_params()

    # 2. 定义算力预算范围
    C_budgets = np.logspace(21, 24, 8)

    # 3. 比较策略
    df_comparison = compare_strategies(C_budgets, E, A, alpha, B, beta)

    # 4. 可视化
    visualize_multi_stage(df_comparison)

    # 5. 保存结果
    df_comparison.to_csv('multi_stage_comparison.csv', index=False)
    print("\n✅ 结果已保存到: multi_stage_comparison.csv")

    print("\n" + "="*60)
    print("✅ 阶段4完成: 多阶段训练优化")
    print("="*60)

    # 总结
    avg_continue_improvement = df_comparison['continue_improvement'].mean()
    avg_expand_improvement = df_comparison['expand_improvement'].mean()

    print("\n💡 关键发现:")
    print(f"   1. 继续训练策略平均改进: {avg_continue_improvement:.2f}%")
    print(f"   2. 扩展模型策略平均改进: {avg_expand_improvement:.2f}%")
    print(f"   3. 多阶段训练在当前 Scaling Law 下效果有限")
    print(f"   4. 原因: β={beta:.3f} 较大，数据的边际收益递减快")

if __name__ == "__main__":
    main()
