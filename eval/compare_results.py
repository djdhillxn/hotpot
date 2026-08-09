import os
import json
import logging
import argparse
import matplotlib.pyplot as plt

from eval.metrics import compute_metrics_by_type, print_segmented_report


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

    b_summary = compute_metrics_by_type(baseline_data)
    r_summary = compute_metrics_by_type(react_data)

    print_segmented_report(b_summary, model_name="Single-Pass RAG (Baseline)")
    print_segmented_report(r_summary, model_name="ReAct Agent (Multi-Hop)")

    b_over = b_summary.get("overall", {})
    r_over = r_summary.get("overall", {})

    info = (
        f"=== COMPARATIVE STUDY: SINGLE-PASS RAG vs REACT AGENT ===\n"
        f"Baseline Dataset Size: {len(baseline_data)} | ReAct Dataset Size: {len(react_data)}\n"
        f"Single-Pass RAG Joint F1: {b_over.get('joint_f1', 0):.2f}% | Latency: {b_over.get('latency', 0):.2f}s | Steps: {b_over.get('steps', 1):.1f}\n"
        f"ReAct Agent Joint F1:     {r_over.get('joint_f1', 0):.2f}% | Latency: {r_over.get('latency', 0):.2f}s | Steps: {r_over.get('steps', 1):.1f}\n"
    )
    print(info)
    logger.info(info)

    metrics_names = ["Ans EM", "Ans F1", "SP EM", "SP F1", "Joint EM", "Joint F1"]
    b_scores = [b_over.get("em", 0), b_over.get("f1", 0), b_over.get("sp_em", 0), b_over.get("sp_f1", 0), b_over.get("joint_em", 0), b_over.get("joint_f1", 0)]
    r_scores = [r_over.get("em", 0), r_over.get("f1", 0), r_over.get("sp_em", 0), r_over.get("sp_f1", 0), r_over.get("joint_em", 0), r_over.get("joint_f1", 0)]

    # Plot Side-by-Side Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    x = range(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar([i - width/2 for i in x], b_scores, width, label="Single-Pass RAG (Baseline)", color="#888888")
    rects2 = ax.bar([i + width/2 for i in x], r_scores, width, label="ReAct Agent (Multi-Hop)", color="#2b5c8f")

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
        f.write("This study compares **Single-Pass RAG (Direct Prompting Baseline)** against the **ReAct Multi-Hop Agent** across overall and segmented question types.\n\n")

        def gain_str(base, act):
            diff = act - base
            rel = (diff / base * 100) if base > 0 else 0
            return f"{diff:+.2f}%", f"{rel:+.2f}%"

        for group in ["overall", "bridge", "comparison"]:
            bg = b_summary.get(group, {})
            rg = r_summary.get(group, {})
            g_label = group.capitalize()

            f.write(f"## {g_label} Question Performance (n = {rg.get('count', 0)})\n\n")
            f.write("| Metric | Single-Pass RAG | ReAct Agent | Absolute Gain | Relative Gain |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: |\n")

            metrics_map = [
                ("Answer EM", "em"),
                ("Answer F1", "f1"),
                ("Supporting Facts EM", "sp_em"),
                ("Supporting Facts F1", "sp_f1"),
                ("Joint EM", "joint_em"),
                ("Joint F1", "joint_f1"),
            ]

            for label, key in metrics_map:
                bv = bg.get(key, 0.0)
                rv = rg.get(key, 0.0)
                abs_g, rel_g = gain_str(bv, rv)
                if "Joint" in label:
                    f.write(f"| **{label}** | **{bv:.2f}%** | **{rv:.2f}%** | **{abs_g}** | **{rel_g}** |\n")
                else:
                    f.write(f"| {label} | {bv:.2f}% | {rv:.2f}% | {abs_g} | {rel_g} |\n")

            b_st = bg.get("steps", 1.0)
            r_st = rg.get("steps", 1.0)
            b_lt = bg.get("latency", 0.0)
            r_lt = rg.get("latency", 0.0)
            f.write(f"| Avg Trajectory Hops | {b_st:.2f} | {r_st:.2f} | {r_st - b_st:+.2f} | N/A |\n")
            f.write(f"| Avg Question Latency | {b_lt:.2f}s | {r_lt:.2f}s | {r_lt - b_lt:+.2f}s | N/A |\n\n")

        f.write("## Visual Metric Comparison Chart\n\n")
        f.write(f"![Comparison Metrics]({os.path.basename(chart_path)})\n\n")

        f.write("## Methodological Observations\n\n")
        f.write("1. **Bridge Questions (Multi-Hop Reasoning)**: Bridge questions require sequential multi-hop retrieval (finding Entity A to discover Entity B). The ReAct agent's iterative reasoning and active memory excel at these multi-turn hops.\n")
        f.write("2. **Comparison Questions (Parallel Retrieval)**: Comparison questions require parallel retrieval of two independent entities. Single-pass RAG struggles when both entities cannot be retrieved in a single query.\n")

    print(f"Generated comparison chart and report in: {output_dir}/")
    print(f"Saved comparison log file to: {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Single-Pass RAG vs ReAct Agent Benchmarks")
    parser.add_argument("--baseline", type=str, default="eval_results/baseline/results.json", help="Path to baseline results JSON")
    parser.add_argument("--react", type=str, default="eval_results/react/results.json", help="Path to ReAct agent results JSON")
    parser.add_argument("--output-dir", type=str, default="eval_results/comparison", help="Output directory for comparison plots and report")

    args = parser.parse_args()
    compare_results(args.baseline, args.react, output_dir=args.output_dir)
