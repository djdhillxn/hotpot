import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

def generate_eval_plots_and_report(results, output_dir="eval_results"):
    os.makedirs(output_dir, exist_ok=True)

    if not results:
        print("No evaluation results to plot.")
        return

    total_count = len(results)
    avg_em = sum(r["exact_match"] for r in results) / total_count
    avg_f1 = sum(r["f1"] for r in results) / total_count
    avg_sp_f1 = sum(r["sp_f1"] for r in results) / total_count
    avg_joint_em = sum(r["joint_em"] for r in results) / total_count
    avg_joint_f1 = sum(r["joint_f1"] for r in results) / total_count
    avg_steps = sum(r["step_count"] for r in results) / total_count
    avg_latency = sum(r.get("latency", 0) for r in results) / total_count

    # 1. Bar Chart of Main Leaderboard Metrics
    plt.figure(figsize=(9, 5))
    metrics_names = ["Ans EM", "Ans F1", "SP F1", "Joint EM", "Joint F1"]
    metrics_values = [avg_em * 100, avg_f1 * 100, avg_sp_f1 * 100, avg_joint_em * 100, avg_joint_f1 * 100]

    palette = sns.color_palette("Blues_d", len(metrics_names))
    bars = plt.bar(metrics_names, metrics_values, color=palette)

    plt.title("HotpotQA FullWiki ReAct Agent Performance (%)", fontsize=14, fontweight="bold")
    plt.ylabel("Score (%)", fontsize=12)
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 2,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "benchmark_metrics.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()

    # 2. Histogram of ReAct Hop Steps
    plt.figure(figsize=(7, 4))
    step_counts = [r["step_count"] for r in results]
    plt.hist(step_counts, bins=range(1, max(step_counts) + 2), align="left", rwidth=0.8, color="#2b5c8f", edgecolor="black")
    plt.title("Distribution of ReAct Reasoning Hops per Question", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Hops", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    hop_chart_path = os.path.join(output_dir, "hop_distribution.png")
    plt.savefig(hop_chart_path, dpi=300)
    plt.close()

    # 3. Generate Markdown Report
    report_path = os.path.join(output_dir, "evaluation_report.md")
    with open(report_path, "w") as f:
        f.write("# HotpotQA FullWiki ReAct Agent Evaluation Report\n\n")
        f.write(f"**Total Questions Evaluated**: {total_count}\n")
        f.write(f"**Average Latency**: {avg_latency:.2f}s per question\n")
        f.write(f"**Average Trajectory Hops**: {avg_steps:.2f} steps\n\n")

        f.write("## Official Leaderboard Summary Metrics\n\n")
        f.write("| Metric | Score (% / Value) |\n")
        f.write("| :--- | :--- |\n")
        f.write(f"| Answer Exact Match (EM) | {avg_em * 100:.1f}% |\n")
        f.write(f"| Answer F1 Score | {avg_f1 * 100:.1f}% |\n")
        f.write(f"| Supporting Facts F1 | {avg_sp_f1 * 100:.1f}% |\n")
        f.write(f"| **Joint Exact Match (Joint EM)** | **{avg_joint_em * 100:.1f}%** |\n")
        f.write(f"| **Joint F1 Score (Joint F1)** | **{avg_joint_f1 * 100:.1f}%** |\n")
        f.write(f"| Average Hops per Question | {avg_steps:.2f} |\n\n")

        f.write("## Evaluation Visualizations\n\n")
        f.write(f"![Benchmark Metrics]({os.path.basename(chart_path)})\n\n")
        f.write(f"![Hop Distribution]({os.path.basename(hop_chart_path)})\n\n")

        f.write("## Sample Question Predictions\n\n")
        for i, r in enumerate(results[:5], 1):
            f.write(f"### Sample {i}\n")
            f.write(f"- **Question**: {r['question']}\n")
            f.write(f"- **Ground Truth**: {r['gold_answer']}\n")
            f.write(f"- **Agent Prediction**: {r['pred_answer']}\n")
            f.write(f"- **Exact Match**: {'PASS' if r['exact_match'] else 'FAIL'}\n")
            f.write(f"- **Joint F1**: {r['joint_f1']:.2f}\n")
            f.write(f"- **Steps**: {r['step_count']}\n\n")

    print(f"Generated evaluation plots and report in directory: {output_dir}/")
