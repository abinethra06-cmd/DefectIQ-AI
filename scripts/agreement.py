"""Compare system decisions with team-labelled reference.csv (indicative only)."""
import argparse
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from storage import load_history  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default=str(ROOT / "reference.csv"))
    args = parser.parse_args()

    reference = pd.read_csv(args.ref)
    required = {"image", "expected"}
    if not required.issubset(reference.columns):
        raise SystemExit("reference.csv must have columns: image,expected")

    latest = load_history(ROOT / "inspections.db").drop_duplicates("image_name")[["image_name", "decision"]]
    merged = reference.merge(latest, left_on="image", right_on="image_name", how="left")
    missing = int(merged["decision"].isna().sum())
    compared = merged.dropna(subset=["decision"])
    manual = int((compared["decision"] == "MANUAL_REVIEW").sum())
    automatic = compared[compared["decision"] != "MANUAL_REVIEW"]
    agreement = (automatic["decision"] == automatic["expected"]).mean() if len(automatic) else 0.0

    print(f"Images compared: {len(compared)} (not yet inspected: {missing})")
    print(f"Sent to manual review: {manual}")
    print(f"Agreement on auto-decided parts: {agreement:.0%} ({len(automatic)} parts)")
    print(pd.crosstab(automatic["expected"], automatic["decision"],
                      rownames=["team"], colnames=["system"]))


if __name__ == "__main__":
    main()
