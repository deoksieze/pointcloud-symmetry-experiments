"""Collapse per-experiment JSON metrics into one summary table.

Run:  python -m src.summarize
"""
import csv
import json
import os

EXPERIMENTS = ["canonical", "rotation_augmented", "rotation_shift"]
METRICS_DIR = os.path.join("outputs", "metrics")


def main():
    rows = []
    for exp in EXPERIMENTS:
        path = os.path.join(METRICS_DIR, f"{exp}.json")
        with open(path) as f:
            m = json.load(f)
        rows.append({
            "experiment": exp,
            "train_rotation": m["config"]["train_rotation"],
            "test_rotation": m["config"]["test_rotation"],
            "final_test_loss": f"{m['final']['test_loss']:.4f}",
            "final_test_acc": f"{m['final']['test_acc']:.4f}",
        })

    out_path = os.path.join(METRICS_DIR, "summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {out_path}")

    print("\nExperiment summary")
    for r in rows:
        print(f"  {r['experiment']:<18} "
              f"train_rot={str(r['train_rotation']):<5} "
              f"test_rot={str(r['test_rotation']):<5} "
              f"test_acc={r['final_test_acc']}")


if __name__ == "__main__":
    main()