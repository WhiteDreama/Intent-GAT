"""
训练趋势可视化工具 (Benchmark Plotter)

【功能说明】
读取由 evaluate.py 生成的 `benchmark.csv` 表格文件，绘制模型性能随训练进度变化的折线图。
主要用于论文绘图或监控训练是否收敛。

【前置条件】
运行 evaluate.py 时必须使用了 `--save_table` 参数。

【终端用法示例】

1. 基础用法:
   python marl_project/plot_benchmark.py --file logs/benchmark.csv

2. 指定输出目录:
   python marl_project/plot_benchmark.py --file logs/benchmark.csv --out logs/my_plots/

【输出结果】
- benchmark_rates.png:  成功率/撞车率/出界率趋势图。
- benchmark_return.png: 平均奖励(Reward)趋势图。
"""
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import os

# 设置绘图风格
try:
    sns.set_theme(style="whitegrid")
except:
    plt.style.use('ggplot')

def plot_benchmark(csv_path, output_dir):
    if not os.path.exists(csv_path):
        print(f"❌ 错误: 找不到文件 {csv_path}")
        return

    print(f"正在读取: {csv_path}")
    df = pd.read_csv(csv_path)

    # 1. 数据预处理
    # 确保 ckpt 是数字 (用于排序)
    # 如果 ckpt 列包含非数字字符，尝试提取数字
    df['ckpt_num'] = pd.to_numeric(df['ckpt'], errors='coerce')
    df = df.sort_values('ckpt_num')

    # 准备输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 2. 绘制：成功率 vs 训练轮数 (Success Rate Curve)
    plt.figure(figsize=(10, 6))
    
    # 转换成百分比
    if df['success'].max() <= 1.0:
        df['success_pct'] = df['success'] * 100
        df['crash_pct'] = df['crash'] * 100
        df['out_pct'] = df['out_of_road'] * 100
    else:
        df['success_pct'] = df['success']
        df['crash_pct'] = df['crash']
        df['out_pct'] = df['out_of_road']

    plt.plot(df['ckpt_num'], df['success_pct'], marker='o', label='Success Rate', linewidth=2, color='green')
    plt.plot(df['ckpt_num'], df['crash_pct'], marker='x', label='Crash Rate', linestyle='--', color='red')
    plt.plot(df['ckpt_num'], df['out_pct'], marker='^', label='Out of Road Rate', linestyle='--', color='orange')

    plt.title("Performance Evolution over Checkpoints")
    plt.xlabel("Checkpoint (Training Steps/Epochs)")
    plt.ylabel("Rate (%)")
    plt.legend()
    plt.grid(True)
    
    out_path = os.path.join(output_dir, "benchmark_rates.png")
    plt.savefig(out_path, dpi=300)
    print(f"✅ 趋势图已保存: {out_path}")

    # 3. 绘制：平均奖励趋势
    plt.figure(figsize=(10, 6))
    # 填充标准差阴影 (如果有 std 列)
    if 'return_std' in df.columns:
        plt.fill_between(df['ckpt_num'], 
                         df['return_mean'] - df['return_std'], 
                         df['return_mean'] + df['return_std'], 
                         color='blue', alpha=0.15)
    
    plt.plot(df['ckpt_num'], df['return_mean'], marker='o', color='blue', label='Mean Return')
    plt.title("Reward Evolution")
    plt.xlabel("Checkpoint")
    plt.ylabel("Episode Return")
    plt.legend()
    
    out_path_ret = os.path.join(output_dir, "benchmark_return.png")
    plt.savefig(out_path_ret, dpi=300)
    print(f"✅ 奖励图已保存: {out_path_ret}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--out", type=str, default="logs/benchmark_plots")
    args = parser.parse_args()
    plot_benchmark(args.file, args.out)