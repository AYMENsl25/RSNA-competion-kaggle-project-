"""Compare saved screening runs without extracting model checkpoints.

Usage: python scripts/compare_saved_experiment_zips.py [results_directory]
The source ZIPs are read only. A versioned comparison is written under results.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna_experiments.comparison import compare_runs  # noqa: E402


def main():
    results = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "results"
    if not results.is_dir():
        raise FileNotFoundError(results)
    archives = sorted(results.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"No saved training ZIPs in {results}")
    with TemporaryDirectory(prefix="rsna_saved_comparison_") as scratch:
        runs = {}
        for archive in archives:
            with ZipFile(archive) as zipped:
                summaries = [name for name in zipped.namelist() if name.endswith("/run_summary.json")]
                if len(summaries) != 1:
                    raise ValueError(f"{archive.name}: expected one run_summary.json")
                summary_name = summaries[0]
                summary = json.loads(zipped.read(summary_name))
                experiment = summary["experiment"]
                prediction_name = summary_name.removesuffix("run_summary.json") + "validation_predictions.csv"
                if prediction_name not in zipped.namelist():
                    raise ValueError(f"{archive.name}: missing validation predictions")
                if experiment in runs:
                    raise ValueError(f"Duplicate saved experiment: {experiment}")
                directory = Path(scratch) / experiment
                directory.mkdir()
                (directory / "run_summary.json").write_bytes(zipped.read(summary_name))
                (directory / "validation_predictions.csv").write_bytes(zipped.read(prediction_name))
                runs[experiment] = directory
        if "r00_resnet_control" not in runs:
            raise ValueError("The matched r00 control output is required")
        ordered = [runs.pop("r00_resnet_control"), *[runs[key] for key in sorted(runs)]]
        output = compare_runs(ordered, results / "screening_comparison", bootstrap=1000)
        receipt = json.loads((output / "comparison.json").read_text())
        print("Comparison:", output)
        print("Validation rows:", receipt["validation_rows"], "groups:", receipt["validation_groups"])
        print("Eligible AUC targets:", len(receipt["eligible_auc_targets"]), "/ 12")
        for row in sorted(receipt["summary"], key=lambda item: item["macro_bce"]):
            print(f"{row['experiment']}: BCE={row['macro_bce']:.6f} AUC={row['macro_auc']:.6f}")
        for pair in receipt["pairwise"]:
            print("vs r00:", pair["run_id"][:12], "delta AUC=", pair["delta_macro_auc"],
                  "95% CI=", (pair["ci_low"], pair["ci_high"]),
                  "valid bootstrap=", pair["bootstrap_valid"])


if __name__ == "__main__":
    main()
