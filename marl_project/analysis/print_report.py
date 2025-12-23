"""
评估报告打印工具 (Report Printer)

【功能说明】
读取由 evaluate.py 生成的 `eval_summary.json`，在终端打印格式化的人类可读报告。
用于快速查看关键指标 (ADE/FDE, 成功率, 风险指标)，而无需打开 JSON 文件慢慢找。

【前置条件】
运行 evaluate.py 时必须使用了 `--save_json` 参数。

【终端用法示例】

1. 打印报告:
   python marl_project/print_report.py --file logs/eval_summary.json

【输出示例】
🔹 模型 1: logs/.../best_model.pth
   [表现]
     ✅ 成功率:     92.00%
     💰 平均分:     450.20
   [预测准确度]
     🎯 ADE: 0.350 m
"""
import json
import argparse
import pandas as pd

def print_pretty_report(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except:
        print("读取失败")
        return

    # 兼容 evaluate.py 可能输出单个字典或字典列表
    if "models" in data:
        items = data["models"]
    else:
        items = [data]

    print("\n" + "="*60)
    print(f" 📄 评估报告: {json_path}")
    print("="*60)

    for i, model in enumerate(items):
        path = model.get("model_path", "Unknown")
        print(f"\n🔹 模型 {i+1}: {path}")
        
        # 提取核心数据
        rates = model.get("rates", {})
        ret = model.get("return", {})
        risk = model.get("risk", {})
        aux = model.get("aux", {})

        # 格式化输出
        print(f"   [表现]")
        print(f"     ✅ 成功率:     {rates.get('success', 0):.2%}")
        print(f"     💥 撞车率:     {rates.get('crash', 0):.2%}")
        print(f"     🚧 出界率:     {rates.get('out_of_road', 0):.2%}")
        print(f"     💰 平均分:     {ret.get('mean', 0):.2f} (±{ret.get('std', 0):.2f})")
        
        print(f"   [风险]")
        print(f"     ⚠️ 最小TTC:    {risk.get('avg_min_ttc_s', 0):.2f} 秒")
        print(f"     📏 最小距离:    {risk.get('avg_min_dist_m', 0):.2f} 米")
        
        print(f"   [预测准确度]")
        print(f"     🎯 ADE (平均误差): {aux.get('ade', 0):.3f} m")
        print(f"     🏁 FDE (终点误差): {aux.get('fde', 0):.3f} m")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()
    print_pretty_report(args.file)