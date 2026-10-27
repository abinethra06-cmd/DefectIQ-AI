import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from severity import load_config, evaluate  # noqa: E402

CONFIG = ROOT / "config.yaml"
OUTSIDE = [0.0, 0.0, 0.1, 0.1]
INSIDE = [0.45, 0.45, 0.55, 0.55]


def defect(kind, area, box, conf=0.9):
    return {
        "type": kind, "confidence": conf, "box": [0, 0, 1, 1],
        "box_norm": box, "area_ratio": area
    }


def test_no_defects_is_accept():
    assert evaluate([], load_config(CONFIG))["decision"] == "ACCEPT"


def test_small_scratch_outside_zone_is_accept():
    assert evaluate([defect("scratches", 0.02, OUTSIDE)], load_config(CONFIG))["decision"] == "ACCEPT"


def test_medium_patch_is_rework():
    assert evaluate([defect("patches", 0.06, OUTSIDE)], load_config(CONFIG))["decision"] == "REWORK"


def test_inclusion_in_critical_zone_is_reject():
    assert evaluate([defect("inclusion", 0.06, INSIDE)], load_config(CONFIG))["decision"] == "REJECT"


def test_low_confidence_goes_to_manual_review():
    assert evaluate([defect("patches", 0.06, OUTSIDE, 0.40)], load_config(CONFIG))["decision"] == "MANUAL_REVIEW"


def test_very_low_confidence_is_ignored():
    assert evaluate([defect("inclusion", 0.06, INSIDE, 0.10)], load_config(CONFIG))["decision"] == "ACCEPT"


def test_worst_defect_decides_the_part():
    result = evaluate([
        defect("scratches", 0.02, OUTSIDE),
        defect("patches", 0.06, OUTSIDE, 0.40),
        defect("inclusion", 0.06, INSIDE),
    ], load_config(CONFIG))
    assert result["decision"] == "REJECT"
    assert "other defect" in result["reason"]


def test_strict_profile_changes_decision_without_retraining():
    sample = [defect("inclusion", 0.05, OUTSIDE)]
    assert evaluate(sample, load_config(CONFIG))["decision"] == "REWORK"
    assert evaluate(sample, load_config(CONFIG, "Strict (critical parts)"))["decision"] == "REJECT"
def test_count_multiplier_is_capped():
    cfg = load_config(CONFIG)
    one = defect("scratches", 0.02, OUTSIDE)
    many = [defect("scratches", 0.02, OUTSIDE) for _ in range(10)]

    one_result = evaluate([one], cfg)
    many_result = evaluate(many, cfg)

    assert many_result["max_score"] <= one_result["max_score"] * 1.25
