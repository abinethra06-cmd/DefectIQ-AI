"""Config-driven severity scoring and disposition decisions."""
import copy
from pathlib import Path
import yaml

DECISION_RANK = {"ACCEPT": 0, "REWORK": 1, "MANUAL_REVIEW": 2, "REJECT": 3}


def load_config(path="config.yaml", profile=None):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if profile and profile in cfg.get("profiles", {}):
        cfg = apply_profile(cfg, cfg["profiles"][profile])
    return cfg


def apply_profile(cfg, overrides):
    cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        if isinstance(value, dict):
            cfg[key] = {**cfg.get(key, {}), **value}
        else:
            cfg[key] = value
    return cfg


def in_critical_zone(box_norm, zone):
    """Check whether the centre of a normalized bounding box is in the zone."""
    if len(box_norm) != 4 or len(zone) != 4:
        raise ValueError("box_norm and critical_zone must each contain four numbers")
    cx = (box_norm[0] + box_norm[2]) / 2
    cy = (box_norm[1] + box_norm[3]) / 2
    return zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]


def score_defect(defect, cfg):
    """Return (severity score 0-100, whether defect centre is critical)."""
    defect_type = defect["type"]
    if defect_type not in cfg["type_weights"]:
        raise ValueError(f"Unknown defect type {defect_type!r}; check model class names")
    area_ratio = float(defect["area_ratio"])
    if not 0 <= area_ratio <= 1:
        raise ValueError("area_ratio must be between 0 and 1")
    full_at = float(cfg["size_full_at"])
    if full_at <= 0:
        raise ValueError("size_full_at must be greater than zero")

    weight = float(cfg["type_weights"][defect_type])
    size_factor = 0.2 + 0.8 * min(1.0, area_ratio / full_at)
    critical = in_critical_zone(defect["box_norm"], cfg["critical_zone"])
    zone_factor = float(cfg["zone_multiplier"]) if critical else 1.0
    score = min(100.0, 100.0 * weight * size_factor * zone_factor)
    return round(score, 1), critical


def decide_defect(score, confidence, cfg):
    if confidence < cfg["review_below"]:
        return "MANUAL_REVIEW"
    if score >= cfg["thresholds"]["reject"]:
        return "REJECT"
    if score >= cfg["thresholds"]["rework"]:
        return "REWORK"
    return "ACCEPT"


def explain(defect, decision):
    zone = "inside critical zone" if defect["critical"] else "outside critical zone"
    message = (
        f"{defect['type']}: {defect['area_ratio'] * 100:.1f}% of surface, "
        f"{zone}, confidence {defect['confidence']:.2f}, "
        f"severity {defect['score']:.1f}/100"
    )
    if decision == "MANUAL_REVIEW":
        message += " (model confidence is low; human check needed)"
    return message


def evaluate(defects, cfg):
    """Return a part-level decision, reason, maximum score, and scored defects."""
    kept = []
    for defect in defects:
        confidence = float(defect["confidence"])
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if confidence >= cfg["min_confidence"]:
            kept.append(defect)

    if not kept:
        return {
            "decision": "ACCEPT",
            "max_score": 0.0,
            "reason": "No defects detected above the confidence threshold.",
            "defects": [],
        }

    scored = []
    for original in kept:
        d = dict(original)
        d["score"], d["critical"] = score_defect(d, cfg)
        d["decision"] = decide_defect(d["score"], d["confidence"], cfg)
        d["explanation"] = explain(d, d["decision"])
        scored.append(d)

    worst = max(scored, key=lambda d: (DECISION_RANK[d["decision"]], d["score"]))
    decision = worst["decision"]
    reason = f"{decision.replace('_', ' ').title()}: {worst['explanation']}"
    if len(scored) > 1:
        reason += f" (+{len(scored) - 1} other defect(s))"

    return {
        "decision": decision,
        "max_score": max(d["score"] for d in scored),
        "reason": reason,
        "defects": scored,
    }
