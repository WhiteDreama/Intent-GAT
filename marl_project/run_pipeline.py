"""
自动化评估与分析流水线 (Automated Evaluation & Analysis Pipeline)

【功能说明】
这是本项目的“总指挥”脚本。它能一键完成从“模型跑分”到“生成图表报告”的全过程。
无需手动分步运行，它会自动串联调用以下工具：
  1. evaluate.py                -> 在环境中运行模型，生成 JSON/CSV 数据。
  2. analysis/analysis_tools.py -> 生成微观分析图 (轨迹热力图、死因分析)。
  3. analysis/plot_benchmark.py -> 生成宏观趋势图 (成功率/奖励随训练变化的折线图)。
  4. analysis/print_report.py   -> 在终端打印最终的文字版成绩单。

【核心参数】
--model_glob: (必填) 模型文件路径模式。支持通配符 '*'。
              示例: "logs/exp/ckpt_*.pth" (批量) 或 "logs/exp/best.pth" (单个)
--out_dir:    (选填) 所有图表和数据的输出目录。默认为 "logs/final_report"。
--episodes:   (选填) 每个模型测试多少局。默认 5。建议: 调试用 5, 论文数据用 50+。
--map:        (选填) 强制指定测试地图。如果不填，则默认使用 config.py 中的训练地图。
              示例: "SCS" (标准考场), "3" (3个随机块), "SCrRT" (复杂路况)。
--render:     (选填) 是否开启 2D 实时渲染窗口 (Top-down view)。

【典型用法示例】

1. [最常用] 批量生成训练趋势图
   评估某次实验的所有存档，看看模型是不是越学越好：
   python marl_project/run_pipeline.py --model_glob "logs/marl_experiment/general/default/ckpt_*.pth" --out_dir "logs/report_training_trend"

2. [交叉评估] 泛化能力测试
   强制使用随机地图 (block_num=3) 来测试之前训练的模型：
   python marl_project/run_pipeline.py --model_glob "logs/marl_experiment/general/default/best_model.pth" --out_dir "logs/report_generalization" --map "3" --episodes 50

3. [基准测试] 基础能力测试
   强制使用标准 SCS 地图进行考试：
   python marl_project/run_pipeline.py --model_glob "logs/marl_experiment/general/default/best_model.pth" --out_dir "logs/report_scs_test" --map "SCS"

4. [视觉调试] 观看驾驶画面
   开启渲染窗口，亲眼看看车是怎么开的：
   python marl_project/run_pipeline.py --model_glob "logs/marl_experiment/general/default/best_model.pth" --render --episodes 5
"""
import os
import argparse
import sys

def run_command(cmd):
    """辅助函数：打印并执行命令，如果出错则停止"""
    print(f"\n🚀 [Pipeline] 执行命令: {cmd}")
    # 使用 sys.executable 确保使用当前的 python 环境
    full_cmd = f"{sys.executable} {cmd}"
    ret = os.system(full_cmd)
    if ret != 0:
        print(f"❌ [Pipeline] 命令执行失败，流程终止。")
        sys.exit(ret)

def main():
    parser = argparse.ArgumentParser(description="一键运行评估与全套可视化分析")
    # 核心参数
    parser.add_argument("--model_glob", type=str, required=True, help="模型路径通配符，例如 'logs/.../ckpt_*.pth'")
    parser.add_argument("--episodes", type=int, default=5, help="每个模型测几局")
    parser.add_argument("--out_dir", type=str, default="logs/final_report", help="所有结果输出到哪里")
    parser.add_argument("--map", type=str, default=None, help="强制指定地图 (例如 S, SCS)")
    
    # === 新增：控制是否开启渲染 ===
    parser.add_argument("--render", action="store_true", help="是否开启可视化窗口")
    
    args = parser.parse_args()

    # 1. 准备路径
    os.makedirs(args.out_dir, exist_ok=True)
    summary_json = os.path.join(args.out_dir, "eval_summary.json")
    details_json = os.path.join(args.out_dir, "eval_details.json")
    benchmark_csv = os.path.join(args.out_dir, "benchmark.csv")
    
    print("="*60)
    print("🤖 自动评估流水线启动")
    print(f"📂 结果输出目录: {args.out_dir}")
    print("="*60)

    # ---------------------------------------------------------
    # 第一步：运行 Evaluate (路径不变，还在 marl_project 下)
    # ---------------------------------------------------------
    print("\n🔹 [Step 1/4] 正在运行评估...")
    
    eval_cmd = (
        f"marl_project/evaluate.py "
        f"--model_glob \"{args.model_glob}\" "
        f"--episodes {args.episodes} "
        f"--save_json \"{summary_json}\" "
        f"--save_details_json \"{details_json}\" "
        f"--save_table \"{benchmark_csv}\" "
    )
    
    if args.map:
        eval_cmd += f" --map_sequence \"{args.map}\""
    
    if args.render:
        eval_cmd += " --render --top_down"

    run_command(eval_cmd)

    # ---------------------------------------------------------
    # 第二步：调用 analysis_tools (路径已更新到 analysis/ 下)
    # ---------------------------------------------------------
    print("\n🔹 [Step 2/4] 正在生成死因与轨迹分析图...")
    analysis_out = os.path.join(args.out_dir, "analysis_charts")
    # ↓↓↓ 注意这里路径变了 ↓↓↓
    run_command(f"marl_project/analysis/analysis_tools.py --file \"{details_json}\" --out \"{analysis_out}\"")

    # ---------------------------------------------------------
    # 第三步：调用 plot_benchmark (路径已更新到 analysis/ 下)
    # ---------------------------------------------------------
    print("\n🔹 [Step 3/4] 正在生成训练趋势对比图...")
    benchmark_out = os.path.join(args.out_dir, "benchmark_plots")
    # ↓↓↓ 注意这里路径变了 ↓↓↓
    run_command(f"marl_project/analysis/plot_benchmark.py --file \"{benchmark_csv}\" --out \"{benchmark_out}\"")

    # ---------------------------------------------------------
    # 第四步：调用 print_report (路径已更新到 analysis/ 下)
    # ---------------------------------------------------------
    print("\n🔹 [Step 4/4] 生成最终文字报告...")
    # ↓↓↓ 注意这里路径变了 ↓↓↓
    run_command(f"marl_project/analysis/print_report.py --file \"{summary_json}\"")

    print("\n" + "="*60)
    print(f"✅ 全流程结束！所有结果已保存在: {args.out_dir}")
    print("="*60)

if __name__ == "__main__":
    main()