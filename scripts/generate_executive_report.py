"""
Generate executive-level visualizations and report.
Creates charts and visual summaries suitable for stakeholder presentations.
Outputs go to docs/images/ and results/metrics/.
"""
import json
import sys
from pathlib import Path


def generate_report(results_dir="results/metrics", output_dir="docs/images"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.2)

    # Load metrics
    comp_path = Path(results_dir) / "comparison_base_vs_trained.json"
    speed_path = Path(results_dir) / "speed_benchmark.json"

    if not comp_path.exists():
        print("⚠️  No comparison metrics found. Run evaluation first.")
        return False

    with open(comp_path) as f:
        comp = json.load(f)
    with open(speed_path) as f:
        speed = json.load(f)

    base = comp["base_model"]
    trained = comp["trained_model"]

    # ── Chart 1: Model Comparison Bar Chart ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy metrics
    metrics = ["mAP50", "mAP50_95", "precision", "recall"]
    labels = ["mAP@50", "mAP@50:95", "Precision", "Recall"]
    base_vals = [base.get(m, 0) for m in metrics]
    trained_vals = [trained.get(m, 0) for m in metrics]

    x = np.arange(len(labels))
    w = 0.35
    axes[0].bar(x - w / 2, base_vals, w, label='Base (Pre-trained)',
                color='#94a3b8', edgecolor='white')
    axes[0].bar(x + w / 2, trained_vals, w, label='Fine-tuned',
                color='#3b82f6', edgecolor='white')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Model Performance Comparison', fontweight='bold', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    for i, (b, t) in enumerate(zip(base_vals, trained_vals)):
        axes[0].text(i - w / 2, b + 0.02, f'{b:.2f}', ha='center',
                     fontsize=9, color='#475569')
        axes[0].text(i + w / 2, t + 0.02, f'{t:.2f}', ha='center',
                     fontsize=9, color='#1e40af')

    # Speed metrics
    fps = speed.get("fps_avg", 0)
    latency = speed.get("latency_avg_ms", 0)
    categories = ['FPS\n(higher=better)', 'Latency (ms)\n(lower=better)']
    values = [fps, latency]
    colors = [
        '#10b981' if fps > 30 else '#f59e0b' if fps > 15 else '#ef4444',
        '#10b981' if latency < 30 else '#f59e0b' if latency < 100 else '#ef4444',
    ]
    bars = axes[1].bar(categories, values, color=colors, edgecolor='white', width=0.5)
    axes[1].set_title('Inference Speed (GPU)', fontweight='bold', fontsize=14)
    for bar, val in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{val:.1f}', ha='center', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/results_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated: {output_dir}/results_chart.png")

    # ── Chart 2: Executive Summary Card ──
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    summary_text = (
        f"EXECUTIVE SUMMARY — YOLOv8 Object Detection\n"
        f"{'━' * 46}\n\n"
        f"  Model:       YOLOv8n (nano) fine-tuned on COCO128\n"
        f"  Accuracy:    mAP@50 = {trained.get('mAP50', 0):.1%}  |  "
        f"mAP@50:95 = {trained.get('mAP50_95', 0):.1%}\n"
        f"  Speed:       {fps:.0f} FPS  |  {latency:.1f}ms per image\n"
        f"  Cost:        $0 (trained on free Google Colab GPU)\n"
        f"  Classes:     80 object categories (COCO standard)\n"
        f"  Export:      PyTorch (.pt) + ONNX (cross-platform)\n\n"
        f"  ✅ Real-time capable  ✅ Production-exportable  ✅ Zero cost"
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontfamily='monospace', fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#f0f9ff',
                      edgecolor='#3b82f6', alpha=0.9))

    plt.savefig(f'{output_dir}/executive_summary.png', dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ Generated: {output_dir}/executive_summary.png")

    return True


if __name__ == "__main__":
    ok = generate_report()
    sys.exit(0 if ok else 1)
