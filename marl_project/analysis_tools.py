import json
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D

# === 配置画图风格 ===
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    pass

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"❌ 错误: 文件不存在 -> {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 读取 JSON 失败: {e}")
        return None

def plot_trajectory(data, output_dir):
    """功能 1: 绘制车辆轨迹图 (支持 evaluate.py 的 details_json 格式)"""
    print("🎨 正在绘制轨迹图...")
    
    # 兼容 evaluate.py 的输出结构
    # 结构通常是: {"episode_summaries": [...], "out_of_road_traces": [...]} 
    # 或者直接是 list of details
    
    # 提取 trace 数据
    traces = []
    
    # 情况 A: 单个模型的 details
    if "episode_summaries" in data:
        # evaluate.py 逻辑里，详细的 step 数据其实没有完全保存下来，
        # 除非你在 evaluate.py 里修改了逻辑或者使用了 out_of_road_traces。
        # 但根据 evaluate.py 源码，它在 'episode_summaries' -> 'out_of_road_traces' 里存了失败轨迹。
        for summary in data.get("episode_summaries", []):
            if "out_of_road_traces" in summary:
                traces.extend(summary["out_of_road_traces"])
    
    # 情况 B: 如果你修改过 evaluate.py 保存了所有 step，这里需要适配
    # 鉴于你的 evaluate.py 源码，目前只能画 "Out of Road" 的最后 30 步轨迹
    
    if not traces:
        print("⚠️  警告: JSON 中没有找到 'out_of_road_traces'。请确保评估时有车辆出界且记录已开启。")
        return

    plt.figure(figsize=(10, 8))
    
    count = 0
    for trace in traces:
        # trace 结构: {'agent_id': '...', 'last_steps': [{'abs_lane_lat': ..., 'speed_kmh': ...}]}
        # 注意：evaluate.py 默认没有存 x,y 坐标 (position)，只存了 lat/heading 等。
        # 如果要画 x,y，必须修改 evaluate.py。
        # 既然没有 x,y，我们画 "横向偏移 (Lane Lateral)" vs "速度" 或者 "时间步"
        
        last_steps = trace.get("last_steps", [])
        if not last_steps: continue
        
        steps = np.arange(len(last_steps))
        lats = [s.get("abs_lane_lat", 0) for s in last_steps]
        speeds = [s.get("speed_kmh", 0) for s in last_steps]
        
        # 画子图：横向偏移变化
        plt.subplot(2, 1, 1)
        plt.plot(steps, lats, alpha=0.5)
        
        # 画子图：速度变化
        plt.subplot(2, 1, 2)
        plt.plot(steps, speeds, alpha=0.5)
        count += 1
        if count > 50: break # 只画前50条防止卡顿

    plt.subplot(2, 1, 1)
    plt.title(f"Out-of-Road Analysis (Last {len(last_steps)} steps)")
    plt.ylabel("Lateral Dist to Center (m)")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.xlabel("Step (Before Crash)")
    plt.ylabel("Speed (km/h)")
    plt.grid(True)

    out_path = os.path.join(output_dir, "failure_analysis.png")
    plt.savefig(out_path, dpi=300)
    print(f"✅ 失败分析图已保存: {out_path}")
    plt.close()

def analyze_statistics(data, output_dir):
    """功能 2: 统计成功率、死因分布和速度"""
    print("📊 正在统计数据...")
    
    summaries = data.get("episode_summaries", [])
    if not summaries:
        print("❌ 数据中没有 'episode_summaries'")
        return

    # 提取所有结局统计
    total_counts = {"success": 0, "crash": 0, "out_of_road": 0, "timeout": 0}
    speeds = []
    
    for summary in summaries:
        counts = summary.get("terminal_counts", {})
        for k in total_counts.keys():
            total_counts[k] += counts.get(k, 0)
        
        if "avg_speed_kmh" in summary:
            speeds.append(summary["avg_speed_kmh"])

    # 1. 画饼图：结局分布
    labels = list(total_counts.keys())
    sizes = list(total_counts.values())
    
    if sum(sizes) > 0:
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#66b3ff','#ff9999','#ffcc99','#99ff99'])
        plt.title("Termination Reasons Distribution")
        plt.savefig(os.path.join(output_dir, "termination_pie.png"))
        plt.close()
        print("✅ 结局饼图已保存")
    
    # 2. 画直方图：平均速度分布
    if speeds:
        plt.figure(figsize=(8, 4))
        plt.hist(speeds, bins=20, color='skyblue', edgecolor='black')
        plt.title(f"Average Speed Distribution (Mean: {np.mean(speeds):.1f} km/h)")
        plt.xlabel("Speed (km/h)")
        plt.ylabel("Count")
        plt.savefig(os.path.join(output_dir, "speed_hist.png"))
        plt.close()
        print("✅ 速度分布图已保存")

def main():
    parser = argparse.ArgumentParser(description="Analyze evaluate.py JSON output")
    parser.add_argument("--file", type=str, required=True, help="Path to the JSON file (e.g., logs/details.json)")
    parser.add_argument("--out", type=str, default="logs/analysis_result", help="Output directory for plots")
    args = parser.parse_args()

    # 1. 准备输出目录
    os.makedirs(args.out, exist_ok=True)
    
    # 2. 加载数据
    data = load_json(args.file)
    if not data: return

    # 3. 执行分析
    # 注意：evaluate.py 生成两种 JSON，一种是 summary (简单)，一种是 details (详细)
    # 这个脚本主要针对 details json (通过 --save_details_json 生成)
    
    if "episode_summaries" in data:
        analyze_statistics(data, args.out)
        plot_trajectory(data, args.out) # 画最后时刻的偏移量
    else:
        print("⚠️  输入的 JSON 似乎是 summary 格式 (缺少 episode_summaries)。")
        print("请在运行 evaluate.py 时使用 --save_details_json 参数以获取可分析的数据。")
        # 对于 summary json，只能打印简单信息
        print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main()