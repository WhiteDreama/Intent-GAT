import os
import subprocess
from datetime import datetime

# === 配置区域 ===
# 存放图片的文件夹名称
OUTPUT_DIR = "analysis_plots_" + datetime.now().strftime("%Y%m%d_%H%M%S")
# 你想要分析的核心 Tag 列表
TARGET_TAGS = [
    "Terminal/SuccessRate",
    "Terminal/OutOfRoadRate",
    "Terminal/CrashRate",
    "Reward/Mean_Step",
    "Reward/Mean_Episode",
    "Loss/Value",
    "Loss/PPO",
    "Loss/Entropy",
    "RewardDecomp/CrashPenalty_per_step",
    "Risk/FracSteps_TTC_Threat"
]

def run_batch():
    # 1. 创建输出文件夹
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")

    # 2. 循环生成每一张图
    for tag in TARGET_TAGS:
        # 格式化文件名：将斜杠替换为下划线，避免路径错误
        safe_name = tag.replace("/", "_")
        out_path = os.path.join(OUTPUT_DIR, f"{safe_name}.pdf")
        
        print(f"正在绘制: {tag} ...")
        
        # 构建命令
        cmd = [
            "python", "plot_paper.py",
            "--tag", tag,
            "--out", out_path,
            "--title", f"Analysis: {tag}"
        ]
        
        # 执行命令
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"绘制 {tag} 失败: {e}")

    print(f"\n全部绘图完成！请查看文件夹: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    run_batch()