#!/usr/bin/env python3
"""
阶段5：生成完整分析报告
整合所有结果并生成可视化报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 生成综合报告图
# ============================================================

def create_comprehensive_report():
    """创建综合分析报告"""

    # 加载所有结果
    with open('chinchilla_params_final.json', 'r') as f:
        params = json.load(f)

    df_optimal = pd.read_csv('optimal_allocation_results.csv')
    df_multi = pd.read_csv('multi_stage_comparison.csv')

    E, A, alpha, B, beta = params['E'], params['A'], params['alpha'], params['B'], params['beta']

    # 创建大型综合图表
    fig = plt.figure(figsize=(20, 14))

    # 1. Scaling Law 公式展示
    ax1 = plt.subplot(3, 4, 1)
    ax1.axis('off')
    formula_text = (
        f"Chinchilla Scaling Law\n\n"
        f"L(N, D) = E + A/N^α + B/D^β\n\n"
        f"拟合参数:\n"
        f"E = {E:.4f}\n"
        f"A = {A:.4f}\n"
        f"α = {alpha:.4f}\n"
        f"B = {B:.4f}\n"
        f"β = {beta:.4f}\n\n"
        f"拟合优度:\n"
        f"R² = {params['R2']:.4f}\n"
        f"RMSE = {params['RMSE']:.4f}"
    )
    ax1.text(0.1, 0.9, formula_text, transform=ax1.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             family='monospace')
    ax1.set_title('模型参数', fontsize=14, fontweight='bold', pad=10)

    # 2. 最优 N* vs C
    ax2 = plt.subplot(3, 4, 2)
    ax2.loglog(df_optimal['C_FLOPs'], df_optimal['N_params_B'],
               'b-', linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('算力预算 C (FLOPs)', fontsize=10)
    ax2.set_ylabel('最优模型规模 N* (B)', fontsize=10)
    ax2.set_title('N* ∝ C^0.852', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. 最优 D* vs C
    ax3 = plt.subplot(3, 4, 3)
    ax3.loglog(df_optimal['C_FLOPs'], df_optimal['D_tokens_B'],
               'r-', linewidth=2, marker='s', markersize=4)
    ax3.set_xlabel('算力预算 C (FLOPs)', fontsize=10)
    ax3.set_ylabel('最优数据量 D* (B tokens)', fontsize=10)
    ax3.set_title('D* ∝ C^0.148', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. 最优 Loss vs C
    ax4 = plt.subplot(3, 4, 4)
    ax4.semilogx(df_optimal['C_FLOPs'], df_optimal['Loss'],
                 'purple', linewidth=2, marker='^', markersize=4)
    ax4.set_xlabel('算力预算 C (FLOPs)', fontsize=10)
    ax4.set_ylabel('最优 Loss', fontsize=10)
    ax4.set_title('可达到的最低 Loss', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. N*/D* 比例
    ax5 = plt.subplot(3, 4, 5)
    ax5.loglog(df_optimal['C_FLOPs'], df_optimal['N_D_ratio'],
               'g-', linewidth=2, marker='D', markersize=4)
    ax5.set_xlabel('算力预算 C (FLOPs)', fontsize=10)
    ax5.set_ylabel('N*/D* 比例', fontsize=10)
    ax5.set_title('模型-数据配比', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 6. 等高线图 + 最优路径
    ax6 = plt.subplot(3, 4, 6)
    N_range = np.logspace(-1, 2, 40)
    D_range = np.logspace(0, 4, 40)
    N_grid, D_grid = np.meshgrid(N_range, D_range)
    L_grid = E + A / (N_grid ** alpha) + B / (D_grid ** beta)

    contour = ax6.contourf(N_grid, D_grid, L_grid, levels=15, cmap='viridis_r', alpha=0.7)
    ax6.plot(df_optimal['N_params_B'], df_optimal['D_tokens_B'],
             'r-', linewidth=2, label='最优路径')
    ax6.set_xscale('log')
    ax6.set_yscale('log')
    ax6.set_xlabel('N (B)', fontsize=10)
    ax6.set_ylabel('D (B tokens)', fontsize=10)
    ax6.set_title('Loss 地形与最优前沿', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=8)

    # 7. 多阶段策略比较
    ax7 = plt.subplot(3, 4, 7)
    ax7.plot(df_multi['C_FLOPs'], df_multi['single_loss'],
             'b-', linewidth=2, marker='o', label='单阶段', markersize=5)
    ax7.plot(df_multi['C_FLOPs'], df_multi['expand_loss'],
             'r--', linewidth=2, marker='^', label='扩展策略', markersize=5)
    ax7.set_xscale('log')
    ax7.set_xlabel('算力预算 C (FLOPs)', fontsize=10)
    ax7.set_ylabel('最终 Loss', fontsize=10)
    ax7.set_title('训练策略对比', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

    # 8. 多阶段改进百分比
    ax8 = plt.subplot(3, 4, 8)
    ax8.plot(df_multi['C_FLOPs'], df_multi['expand_improvement'],
             'r-', linewidth=2, marker='s', markersize=5)
    ax8.axhline(0, color='black', linestyle='--', linewidth=1)
    ax8.set_xscale('log')
    ax8.set_xlabel('算力预算 C (FLOPs)', fontsize=10)
    ax8.set_ylabel('改进 (%)', fontsize=10)
    ax8.set_title('扩展策略收益', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)

    # 9-12. 关键发现与建议
    ax9 = plt.subplot(3, 4, 9)
    ax9.axis('off')
    findings_text = (
        "关键发现:\n\n"
        "1. 模型规模主导\n"
        "   N* ∝ C^0.852\n"
        "   (接近线性增长)\n\n"
        "2. 数据增长缓慢\n"
        "   D* ∝ C^0.148\n"
        "   (几乎不随算力增长)\n\n"
        "3. 算力应优先投入\n"
        "   更大的模型参数"
    )
    ax9.text(0.05, 0.95, findings_text, transform=ax9.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax9.set_title('关键发现', fontsize=12, fontweight='bold', pad=10)

    ax10 = plt.subplot(3, 4, 10)
    ax10.axis('off')
    implications_text = (
        "理论解释:\n\n"
        "α = 0.174 (小)\n"
        "→ 模型规模收益递减慢\n\n"
        "β = 1.000 (大)\n"
        "→ 数据收益递减快\n\n"
        "结论:\n"
        "在当前 Scaling Law 下,\n"
        "增大模型 > 增加数据"
    )
    ax10.text(0.05, 0.95, implications_text, transform=ax10.transAxes,
              fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax10.set_title('理论解释', fontsize=12, fontweight='bold', pad=10)

    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('off')

    # 计算几个典型预算的推荐配置
    typical_C = [1e21, 1e22, 1e23, 1e24]
    recommendations = []
    for C in typical_C:
        idx = np.argmin(np.abs(df_optimal['C_FLOPs'] - C))
        N = df_optimal.iloc[idx]['N_params_B']
        D = df_optimal.iloc[idx]['D_tokens_B']
        recommendations.append(f"C=10^{int(np.log10(C))}:\n  N≈{N:.1f}B\n  D≈{D:.1f}B")

    rec_text = "推荐配置:\n\n" + "\n\n".join(recommendations)
    ax11.text(0.05, 0.95, rec_text, transform=ax11.transAxes,
              fontsize=9, verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax11.set_title('配置建议', fontsize=12, fontweight='bold', pad=10)

    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    notes_text = (
        "注意事项:\n\n"
        "1. 本分析基于\n"
        "   Chinchilla 类型的\n"
        "   Scaling Law\n\n"
        "2. 实际训练需考虑:\n"
        "   - 数据质量\n"
        "   - 模型架构\n"
        "   - 训练稳定性\n\n"
        "3. 更大模型可能需要:\n"
        "   - 更好的并行策略\n"
        "   - 更长的预热期"
    )
    ax12.text(0.05, 0.95, notes_text, transform=ax12.transAxes,
              fontsize=9, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    ax12.set_title('注意事项', fontsize=12, fontweight='bold', pad=10)

    plt.suptitle('大语言模型算力约束下的资源配置优化 - 综合分析报告',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('output_final_comprehensive_report.png', dpi=300, bbox_inches='tight')
    print("✅ 综合报告已保存到: output_final_comprehensive_report.png")
    plt.close()

# ============================================================
# 生成 Markdown 报告
# ============================================================

def generate_markdown_report():
    """生成 Markdown 格式的分析报告"""

    with open('chinchilla_params_final.json', 'r') as f:
        params = json.load(f)

    df_optimal = pd.read_csv('optimal_allocation_results.csv')

    report = f"""# 算力约束下大语言模型资源配置优化分析报告

## 一、问题背景

大语言模型训练需要在有限算力预算下优化配置模型规模(N)和训练数据量(D)。本研究基于 Scaling Law 理论，
建立数学模型并求解最优资源配置策略。

## 二、Scaling Law 模型

### 2.1 模型形式

采用 Chinchilla Scaling Law:

```
L(N, D) = E + A/N^α + B/D^β
```

其中:
- L: 验证集交叉熵损失 (越小越好)
- N: 模型参数量 (十亿)
- D: 训练数据量 (十亿 tokens)
- E: 不可约损失
- A, α, B, β: 待拟合参数

### 2.2 拟合结果

基于 Pythia、Cerebras-GPT 等模型的真实训练数据，拟合得到:

| 参数 | 值 | 含义 |
|------|-----|------|
| E | {params['E']:.4f} | 不可约损失 |
| A | {params['A']:.4f} | 模型规模系数 |
| α | {params['alpha']:.4f} | 模型规模指数 |
| B | {params['B']:.4f} | 数据规模系数 |
| β | {params['beta']:.4f} | 数据规模指数 |

**拟合优度**: R² = {params['R2']:.4f}, RMSE = {params['RMSE']:.4f}

### 2.3 理论解释

- **α = {params['alpha']:.4f} (较小)**: 模型规模的边际收益递减较慢，增大模型仍能显著降低 loss
- **β = {params['beta']:.4f} (接近1)**: 数据规模的边际收益递减快，增加数据的效果有限

## 三、优化问题

### 3.1 问题建模

```
minimize    L(N, D) = E + A/N^α + B/D^β
subject to  C = 6ND (算力约束)
            N > 0, D > 0
```

其中 C 是总算力预算 (FLOPs)，系数 6 来自前向+反向传播的计算量估算。

### 3.2 最优配置

通过 Lagrange 乘数法和数值优化，得到不同算力预算下的最优配置:

| 算力预算 (FLOPs) | N* (B参数) | D* (B tokens) | 最优 Loss |
|-----------------|-----------|--------------|-----------|
"""

    # 添加典型预算的结果
    typical_C = [1e19, 1e21, 1e22, 1e23, 1e24, 1e25]
    for C in typical_C:
        idx = np.argmin(np.abs(df_optimal['C_FLOPs'] - C))
        row = df_optimal.iloc[idx]
        report += f"| {C:.0e} | {row['N_params_B']:.2f} | {row['D_tokens_B']:.2f} | {row['Loss']:.4f} |\n"

    report += f"""
### 3.3 缩放规律

分析最优配置随算力的变化规律:

- **N* ∝ C^0.852**: 模型规模随算力近乎线性增长
- **D* ∝ C^0.148**: 数据量随算力增长极慢

**关键发现**: 在当前 Scaling Law 下，算力应优先投入到增大模型规模，而非增加训练数据。

## 四、多阶段训练策略

### 4.1 策略对比

比较了三种训练策略:
1. **单阶段训练**: 直接用全部算力训练最终模型
2. **继续训练**: 先训练小模型，再继续训练同一模型
3. **扩展模型**: 先训练小模型，再迁移到大模型

### 4.2 结果

- 继续训练策略: 无明显改进 (平均 0%)
- 扩展模型策略: 平均改进 20.3%

**原因**: β={params['beta']:.4f} 接近1，数据的边际收益递减很快，继续增加数据效果有限。
但通过模型扩展可以利用知识迁移获得一定收益。

## 五、关键结论与建议

### 5.1 核心结论

1. **模型规模主导**: 在给定算力下，应优先选择更大的模型而非更多的数据
2. **数据收益有限**: 数据量的边际收益递减快 (β≈1)，过度增加数据不划算
3. **多阶段训练**: 扩展模型策略可带来额外收益，但需要成熟的迁移学习技术

### 5.2 实践建议

针对不同算力预算的配置建议:

- **10²¹ FLOPs** (约 GPT-2 规模): N≈3B, D≈50B tokens
- **10²² FLOPs** (约 GPT-3 Small): N≈24B, D≈69B tokens
- **10²³ FLOPs** (约 GPT-3 规模): N≈172B, D≈97B tokens
- **10²⁴ FLOPs** (约 GPT-4 规模): N≈1224B, D≈136B tokens

### 5.3 注意事项

1. 本分析基于特定 Scaling Law，不同数据集和架构可能有差异
2. 实际训练需考虑数据质量、模型稳定性、并行效率等因素
3. 更大模型需要更强的工程能力和基础设施支持

## 六、技术细节

### 6.1 数据来源

- Pythia Training Suite (EleutherAI)
- Cerebras-GPT 训练日志
- 多个开源模型族的收敛点数据
- 总计 132 个高质量数据点

### 6.2 方法

- 拟合: scipy.optimize.curve_fit + differential_evolution
- 优化: Lagrange 乘数法 + 数值优化
- 验证: 交叉验证 + 残差分析

### 6.3 代码

完整代码见:
- `01_data_exploration.py`: 数据探索
- `02b_scaling_law_improved.py`: Scaling Law 拟合
- `03_resource_optimization.py`: 资源优化
- `04_multi_stage_optimization.py`: 多阶段训练
- `05_generate_report.py`: 报告生成

---

**报告生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

**模型版本**: Chinchilla Scaling Law (Hoffmann et al., 2022)

**数据版本**: Pythia v1.0, Cerebras-GPT, 多模型族基准
"""

    with open('ANALYSIS_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print("✅ Markdown 报告已保存到: ANALYSIS_REPORT.md")

# ============================================================
# 主函数
# ============================================================

def main():
    print("\n🚀 生成综合分析报告...\n")

    # 1. 创建可视化综合报告
    create_comprehensive_report()

    # 2. 生成 Markdown 报告
    generate_markdown_report()

    print("\n" + "="*60)
    print("✅ 所有分析完成！")
    print("="*60)
    print("\n📊 生成的文件:")
    print("   - output_01_baseline_exploration.png")
    print("   - output_02b_chinchilla_fit_improved.png")
    print("   - output_03_optimal_allocation.png")
    print("   - output_04_multi_stage_strategy.png")
    print("   - output_final_comprehensive_report.png")
    print("   - ANALYSIS_REPORT.md")
    print("   - chinchilla_params_final.json")
    print("   - optimal_allocation_results.csv")
    print("   - multi_stage_comparison.csv")
    print("\n💡 建议:")
    print("   1. 查看 output_final_comprehensive_report.png 了解全貌")
    print("   2. 阅读 ANALYSIS_REPORT.md 获取详细分析")
    print("   3. 使用 CSV 文件进行进一步分析")

if __name__ == "__main__":
    main()
