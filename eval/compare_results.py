import os
import json
import logging
import argparse
import matplotlib.pyplot as plt

def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "comparison.log")

    logger = logging.getLogger("compare_runner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode="w")
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, log_file


def compare_results(baseline_json_path, react_json_path, output_dir="eval_results/comparison"):
    logger, log_file = setup_logger(output_dir)

    if not os.path.exists(baseline_json_path):
        err = f"Error: Baseline results file not found at '{baseline_json_path}'. Run `python eval/run_baseline.py` first."
        print(err)
        logger.error(err)
        return
    if not os.path.exists(react_json_path):
        err = f"Error: ReAct Agent results file not found at '{react_json_path}'. Run `python eval/run_eval.py` first."
        print(err)
        logger.error(err)
        return

    with open(baseline_json_path, "r") as f:
        baseline_data = json.load(f)

    with open(react_json_path, "r") as f:
        react_data = json.load(f)

    def calc_metrics(data):
        count = len(data)
        if count == 0:
            return 0, 0, 0, 0, 0, 0, 0, 0
        em = sum(r["exact_match"] for r in data) / count
        f1 = sum(r["f1"] for r in data) / count
        sp_em = sum(r["sp_em"] for r in data) / count
        sp_f1 = sum(r["sp_f1"] for r in data) / count
        joint_em = sum(r["joint_em"] for r in data) / count
        joint_f1 = sum(r["joint_f1"] for r in data) / count
        steps = sum(r.get("step_count", 1) for r in data) / count
        lat = sum(r.get("latency", 0) for r in data) / count
        return em * 100, f1 * 100, sp_em * 100, sp_f1 * 100, joint_em * 100, joint_f1 * 100, steps, lat

    b_em, b_f1, b_sp_em, b_sp_f1, b_joint_em, b_joint_f1, b_steps, b_lat = calc_metrics(baseline_data)
    r_em, r_f1, r_sp_em, r_sp_f1, r_joint_em, r_joint_f1, r_steps, r_lat = calc_metrics(react_data)

    info = (
        f"=== COMPARATIVE STUDY: SINGLE-PASS RAG vs REACT AGENT ===\n"
        f"Baseline Dataset Size: {len(baseline_data)} | ReAct Dataset Size: {len(react_data)}\n"
        f"Single-Pass RAG Joint F1: {b_joint_f1:.2f}% | Latency: {b_lat:.2f}s | Steps: {b_steps:.1f}\n"
        f"ReAct Agent Joint F1:     {r_joint_f1:.2f}% | Latency: {r_lat:.2f}s | Steps: {r_steps:.1f}\n"
    )
    print(info)
    logger.info(info)

    metrics_names = ["Ans EM", "Ans F1", "SP EM", "SP F1", "Joint EM", "Joint F1"]
    baseline_scores = [b_em, b_f1, b_sp_em, b_sp_f1, b_joint_em, b_joint_f1]
    react_scores = [r_em, r_f1, r_sp_em, r_sp_f1, r_joint_em, r_joint_f1]

    # Plot Side-by-Side Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    x = range(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar([i - width/2 for i in x], baseline_scores, width, label="Single-Pass RAG (Baseline)", color="#888888")
    rects2 = ax.bar([i + width/2 for i in x], react_scores, width, label="ReAct Agent (Multi-Hop)", color="#2b5c8f")

    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("HotpotQA FullWiki: Single-Pass RAG vs ReAct Multi-Hop Agent", fontsize=14, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontweight="bold", fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "comparison_metrics.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()

    # Generate Markdown Comparison Report
    report_path = os.path.join(output_dir, "comparison_report.md")
    with open(report_path, "w") as f:
        f.write("# HotpotQA FullWiki: Single-Pass RAG vs ReAct Agent Study\n\n")
        f.write("This study compares **Single-Pass RAG (Direct Prompting Baseline)** against the **ReAct Multi-Hop Agent**.\n\n")

        f.write("## Quantitative Metric Comparison\n\n")
        f.write("| Metric | Single-Pass RAG | ReAct Agent | Absolute Gain | Relative Improvement |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")

        def gain_str(base, act):
            diff = act - base
            rel = (diff / base * 100) if base > 0 else 0
            return f"{diff:+.1f}%", f"{rel:+.1f}%"

        g_em, r_em_str = gain_str(b_em, r_em)
        g_f1, r_f1_str = gain_str(b_f1, r_f1)
        g_sp_em, r_sp_em_str = gain_str(b_sp_em, r_sp_em)
        g_sp, r_sp_str = gain_str(b_sp_f1, r_sp_f1)
        g_jem, r_jem_str = gain_str(b_joint_em, r_joint_em)
        g_jf1, r_jf1_str = gain_str(b_joint_f1, r_joint_f1)

        f.write(f"| Answer EM | {b_em:.1f}% | {r_em:.1f}% | {g_em} | {r_em_str} |\n")
        f.write(f"| Answer F1 | {b_f1:.1f}% | {r_f1:.1f}% | {g_f1} | {r_f1_str} |\n")
        f.write(f"| Supporting Facts EM | {b_sp_em:.1f}% | {r_sp_em:.1f}% | {g_sp_em} | {r_sp_em_str} |\n")
        f.write(f"| Supporting Facts F1 | {b_sp_f1:.1f}% | {r_sp_f1:.1f}% | {g_sp} | {r_sp_str} |\n")
        f.write(f"| **Joint EM** | **{b_joint_em:.1f}%** | **{r_joint_em:.1f}%** | **{g_jem}** | **{r_jem_str}** |\n")
        f.write(f"| **Joint F1** | **{b_joint_f1:.1f}%** | **{r_joint_f1:.1f}%** | **{g_jf1}** | **{r_jf1_str}** |\n")
        f.write(f"| Avg Trajectory Hops | {b_steps:.2f} | {r_steps:.2f} | {r_steps - b_steps:+.2f} | N/A |\n")
        f.write(f"| Avg Question Latency | {b_lat:.2f}s | {r_lat:.2f}s | {r_lat - b_lat:+.2f}s | N/A |\n\n")

        f.write("## Visual Metric Comparison Chart\n\n")
        f.write(f"![Comparison Metrics]({os.path.basename(chart_path)})\n\n")

        f.write("## Methodological Observations\n\n")
        f.write("1. **Single-Pass Retrieval Bottleneck**: Single-pass RAG relies on a single retrieval query, which limits passage coverage when resolving multi-hop bridge entities.\n")
        f.write("2. **Iterative Multi-Hop Retrieval**: The ReAct agent alternates between reasoning and targeted retrieval actions (`search` and `lookup`), allowing intermediate evidence discovery across multiple turns.\n")

    print(f"Generated comparison chart and report in: {output_dir}/")
    print(f"Saved comparison log file to: {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Single-Pass RAG vs ReAct Agent Benchmarks")
    parser.add_argument("--baseline", type=str, default="eval_results/baseline/results.json", help="Path to baseline results JSON")
    parser.add_argument("--react", type=str, default="eval_results/react/results.json", help="Path to ReAct agent results JSON")
    parser.add_argument("--output-dir", type=str, default="eval_results/comparison", help="Output directory for comparison plots and report")

    args = parser.parse_args()
    compare_results(args.baseline, args.react, output_dir=args.output_dir)
