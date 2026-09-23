#!/usr/bin/env python3
"""
阶段1：数据探索与预处理
探索 Scaling Laws 核心数据集
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 数据路径
DATA_DIR = Path("real_attachments/B_scaling_laws")

def load_baseline_data():
    """加载基准收敛数据"""
    df = pd.read_csv(DATA_DIR / "scaling_baseline.csv")
    print("=" * 60)
    print("📊 Scaling Baseline Data (收敛点数据)")
    print("=" * 60)
    print(f"数据形状: {df.shape}")
    print(f"\n列名: {df.columns.tolist()}")
    print(f"\n数据类型:\n{df.dtypes}")
    print(f"\n基本统计:\n{df.describe()}")
    print(f"\n缺失值:\n{df.isnull().sum()}")
    print(f"\n模型家族分布:\n{df['family'].value_counts()}")
    return df

def load_pythia_training_log():
    """加载 Pythia 训练日志"""
    df = pd.read_csv(DATA_DIR / "pythia_training_log_existing.csv")
    print("\n" + "=" * 60)
    print("📊 Pythia Training Log (训练轨迹)")
    print("=" * 60)
    print(f"数据形状: {df.shape}")
    print(f"\n前5列: {df.columns[:5].tolist()}")
    print(f"\n数据预览:\n{df.head()}")
    return df

def load_cerebras_training_log():
    """加载 Cerebras 训练日志"""
    df = pd.read_csv(DATA_DIR / "cerebras_training_log.csv")
    print("\n" + "=" * 60)
    print("📊 Cerebras Training Log")
    print("=" * 60)
    print(f"数据形状: {df.shape}")
    print(f"\n列名: {df.columns.tolist()}")
    print(f"\n模型参数量范围: {df['N_params_B'].min():.3f}B - {df['N_params_B'].max():.3f}B")
    print(f"训练数据量范围: {df['D_tokens_B'].min():.1f}B - {df['D_tokens_B'].max():.1f}B tokens")
    return df

def explore_training_trajectories():
    """探索训练轨迹文件"""
    traj_dir = DATA_DIR / "training_trajectories"
    traj_files = list(traj_dir.glob("*.csv"))

    print("\n" + "=" * 60)
    print(f"📊 Training Trajectories (共{len(traj_files)}个模型)")
    print("=" * 60)

    trajectories = {}
    for file in sorted(traj_files):
        model_size = file.stem.replace("pythia_", "").replace("_trajectory", "")
        df = pd.read_csv(file)
        trajectories[model_size] = df
        print(f"\n{model_size}B: {len(df)} 个checkpoint")
        if 'step' in df.columns and 'val_loss' in df.columns:
            print(f"  Loss范围: {df['val_loss'].min():.4f} - {df['val_loss'].max():.4f}")

    return trajectories

def visualize_baseline_data(df):
    """可视化基准数据"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. N vs Loss
    ax = axes[0, 0]
    for family in df['family'].unique():
        family_data = df[df['family'] == family]
        ax.scatter(family_data['N_params_B'], family_data['val_loss'],
                  label=family, alpha=0.7, s=100)
    ax.set_xscale('log')
    ax.set_xlabel('Model Parameters N (Billions)', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Model Size vs Loss', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. D vs Loss
    ax = axes[0, 1]
    for family in df['family'].unique():
        family_data = df[df['family'] == family]
        ax.scatter(family_data['D_tokens_B'], family_data['val_loss'],
                  label=family, alpha=0.7, s=100)
    ax.set_xscale('log')
    ax.set_xlabel('Training Tokens D (Billions)', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Data Size vs Loss', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 3. N vs D (散点图)
    ax = axes[1, 0]
    scatter = ax.scatter(df['N_params_B'], df['D_tokens_B'],
                        c=df['val_loss'], cmap='viridis_r',
                        s=100, alpha=0.7, edgecolors='black')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Model Parameters N (Billions)', fontsize=12)
    ax.set_ylabel('Training Tokens D (Billions)', fontsize=12)
    ax.set_title('N-D Space (colored by Loss)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Validation Loss')
    ax.grid(True, alpha=0.3)

    # 4. 算力分布 (C = 6ND)
    ax = axes[1, 1]
    df['compute_FLOPs'] = 6 * df['N_params_B'] * df['D_tokens_B'] * 1e18
    ax.hist(np.log10(df['compute_FLOPs']), bins=20, alpha=0.7, edgecolor='black')
    ax.set_xlabel('log10(Compute FLOPs)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Training Compute Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output_01_baseline_exploration.png', dpi=300, bbox_inches='tight')
    print("\n✅ 可视化已保存到: output_01_baseline_exploration.png")
    plt.close()

def main():
    print("\n🚀 开始数据探索...\n")

    # 1. 加载基准数据
    baseline_df = load_baseline_data()

    # 2. 加载训练日志
    pythia_log = load_pythia_training_log()
    cerebras_log = load_cerebras_training_log()

    # 3. 探索训练轨迹
    trajectories = explore_training_trajectories()

    # 4. 可视化
    visualize_baseline_data(baseline_df)

    # 5. 保存清洗后的数据
    baseline_df.to_csv('processed_baseline.csv', index=False)
    print("\n✅ 清洗后的基准数据已保存到: processed_baseline.csv")

    print("\n" + "=" * 60)
    print("✅ 阶段1完成：数据探索与预处理")
    print("=" * 60)

if __name__ == "__main__":
    main()
