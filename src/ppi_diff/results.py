from __future__ import annotations

from pathlib import Path


def maybe_generate_plots(
    output_dir: str | Path,
    aggregate_rows: list[dict[str, object]],
    pr_curves: dict[str, list[tuple[float, float]]],
    score_distribution: dict[str, list[float]] | None,
    k_values: list[int],
) -> dict[str, object]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        return {
            "enabled": False,
            "engine": None,
            "reason": "matplotlib is not installed; skipped PNG plot generation.",
        }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    recall_plot_path = output_path / "recall_at_k.png"
    pr_curve_path = output_path / "pr_curve.png"
    score_distribution_path = output_path / "score_distribution.png"

    _plot_recall_at_k(aggregate_rows, k_values, recall_plot_path, plt)
    _plot_pr_curve(pr_curves, pr_curve_path, plt)
    if score_distribution is not None:
        _plot_score_distribution(score_distribution, score_distribution_path, plt)

    generated = [str(recall_plot_path), str(pr_curve_path)]
    if score_distribution is not None:
        generated.append(str(score_distribution_path))

    return {
        "enabled": True,
        "engine": "matplotlib",
        "generated": generated,
    }


def _plot_recall_at_k(
    aggregate_rows: list[dict[str, object]],
    k_values: list[int],
    output_path: Path,
    plt,
) -> None:
    plt.figure(figsize=(7, 5))
    for row in aggregate_rows:
        method = str(row["method"])
        recalls = [float(row.get(f"recall_at_{k}_mean", float("nan"))) for k in k_values]
        plt.plot(k_values, recalls, marker="o", label=method)
    plt.xlabel("k")
    plt.ylabel("Recall@k")
    plt.title("Recall@k Across Methods")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_pr_curve(pr_curves: dict[str, list[tuple[float, float]]], output_path: Path, plt) -> None:
    plt.figure(figsize=(6, 6))
    for method, curve in pr_curves.items():
        if not curve:
            continue
        recalls = [point[0] for point in curve]
        precisions = [point[1] for point in curve]
        plt.step(recalls, precisions, where="post", label=method)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_score_distribution(
    score_distribution: dict[str, list[float]],
    output_path: Path,
    plt,
) -> None:
    positives = score_distribution.get("test_hits", [])
    negatives = score_distribution.get("other_candidates", [])

    plt.figure(figsize=(7, 5))
    if negatives:
        plt.hist(negatives, bins=30, alpha=0.6, density=True, label="Other candidates")
    if positives:
        plt.hist(positives, bins=30, alpha=0.6, density=True, label="Test hits")
    plt.xlabel("PPR score")
    plt.ylabel("Density")
    plt.title("PPR Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
