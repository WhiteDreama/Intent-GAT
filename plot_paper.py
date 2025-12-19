import argparse
import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

try:
    import scienceplots  # noqa: F401
    _HAS_SCIENCEPLOTS = True
except Exception:
    _HAS_SCIENCEPLOTS = False

# === 全局配置默认值 ===
DEFAULT_FIG_SIZE = (6, 4)
DEFAULT_DPI = 300
SMOOTH_WINDOW = 50

def find_latest_log(log_dir: str = "."):
    """自动寻找目录下最新的 tfevents 文件"""
    files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents*"), recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def list_scalar_tags(log_file: str):
    """列出 tfevents 里所有 scalar tags"""
    ea = EventAccumulator(log_file)
    ea.Reload()
    tags = ea.Tags()
    return list(tags.get("scalars", []))

def extract_data(log_file, tag):
    """从日志提取数据"""
    print(f"读取日志: {log_file}")
    try:
        ea = EventAccumulator(log_file)
        ea.Reload()
        
        # 模糊匹配 Tag (防止 Tag 写法不一致)
        available_tags = list(ea.Tags().get("scalars", []))
        target_tag = tag
        
        if tag not in available_tags:
            # 尝试自动修正，比如用户输了 'reward' 但日志里是 'EpReward'
            matches = [t for t in available_tags if tag.lower() in t.lower()]
            if matches:
                target_tag = matches[0]
                print(f"未找到精确 Tag '{tag}'，自动匹配为: '{target_tag}'")
            else:
                print(f"错误: Tag '{tag}' 不存在。可用 Tags: {available_tags}")
                return None

        scalars = ea.Scalars(target_tag)
        df = pd.DataFrame({
            "Step": [s.step for s in scalars],
            "Value": [s.value for s in scalars]
        })
        return df, target_tag
    except Exception as e:
        print(f"读取失败: {e}")
        return None, None

def plot_data(df, tag_name, output_file, title):
    """绘图逻辑"""
    # 尝试应用学术风格，如果没装 Latex 则回退
    if _HAS_SCIENCEPLOTS:
        try:
            plt.style.use(['science', 'ieee', 'no-latex'])
        except Exception:
            sns.set_theme(style="whitegrid")
            print("SciencePlots 风格加载失败，降级为 Seaborn 默认风格")
    else:
        sns.set_theme(style="whitegrid")

    plt.figure(figsize=DEFAULT_FIG_SIZE)

    # 1. 原始数据 (半透明背景)
    sns.lineplot(data=df, x="Step", y="Value", alpha=0.15, color='grey', label='Raw')

    # 2. 平滑数据 (主线)
    df['Smoothed'] = df['Value'].rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
    sns.lineplot(data=df, x="Step", y="Smoothed", linewidth=1.5, label=f'Smoothed ({tag_name})')

    # 3. 装饰
    plt.xlabel("Training Steps", fontsize=10)
    plt.ylabel(tag_name, fontsize=10)
    plt.title(title if title else f"Training Curve: {tag_name}", fontsize=12)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.3)

    # 4. 保存
    plt.savefig(output_file, dpi=DEFAULT_DPI, bbox_inches='tight')
    print(f"图表已生成: {output_file}")
    # plt.show() # 如果在服务器上运行，注释掉这行

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-Plotter for RL Papers")
    parser.add_argument("--log_dir", type=str, default=".", help="Search directory for tfevents when --file is not set.")
    parser.add_argument("--file", type=str, default=None, help="Path to tfevents file. Defaults to latest under --log_dir.")
    parser.add_argument("--list_tags", action="store_true", help="List available scalar tags and exit.")
    parser.add_argument(
        "--tag",
        type=str,
        default="Reward/Mean_Episode",
        help="Scalar tag to plot (e.g., Reward/Mean_Episode, Terminal/CrashRate, Loss/Total).",
    )
    parser.add_argument("--out", type=str, default="paper_plot.pdf", help="Output filename.")
    parser.add_argument("--title", type=str, default=None, help="Chart title.")
    
    args = parser.parse_args()

    # 1. 确定文件
    target_file = args.file
    if target_file is None:
        target_file = find_latest_log(args.log_dir)
        if target_file is None:
            print("当前目录下找不到任何日志文件！")
            raise SystemExit(1)
        print(f"自动锁定最新日志: {target_file}")
    elif os.path.isdir(target_file):
        # allow passing a directory to --file
        candidate = find_latest_log(target_file)
        if candidate is None:
            print(f"目录下找不到任何日志文件: {target_file}")
            raise SystemExit(1)
        target_file = candidate
        print(f"自动锁定最新日志: {target_file}")

    if args.list_tags:
        tags = list_scalar_tags(target_file)
        print("Available scalar tags:")
        for t in tags:
            print(t)
        raise SystemExit(0)

    # 2. 提取与绘图
    df, final_tag = extract_data(target_file, args.tag)
    if df is not None:
        plot_data(df, final_tag, args.out, args.title)