
from pathlib import Path
import random
import shutil
import yaml

# Project and source paths
PROJECT = Path(__file__).resolve().parents[1]
NEU = PROJECT / "data" / "neu_yolo"
GC10 = Path.home() / "Downloads" / "GC10-DET" / "data"
OUT = PROJECT / "data" / "steel_combined_v1"

SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

NEU_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

GC10_NAMES = [
    "crease",
    "crescent_gap",
    "inclusion",
    "oil_spot",
    "punching_hole",
    "rolled_pit",
    "silk_spot",
    "waist_folding",
    "water_spot",
    "welding_line",
]

# Keep datasets separate if output already exists.
if OUT.exists():
    raise FileExistsError(
        f"{OUT} already exists. Rename or back it up before rerunning."
    )

if not (NEU / "data.yaml").exists():
    raise FileNotFoundError(f"Missing NEU config: {NEU / 'data.yaml'}")

if not (GC10 / "data.yaml").exists():
    raise FileNotFoundError(f"Missing GC10 config: {GC10 / 'data.yaml'}")

with (GC10 / "data.yaml").open(encoding="utf-8") as f:
    gc_config = yaml.safe_load(f)

gc_names_raw = gc_config.get("names", {})
if isinstance(gc_names_raw, dict):
    gc_names = [gc_names_raw[i] for i in sorted(gc_names_raw)]
else:
    gc_names = list(gc_names_raw)

if gc_names != GC10_NAMES:
    raise ValueError(
        "GC10 class list differs from the expected list.\n"
        f"Found: {gc_names}"
    )

# Inclusion is shared by name, but the source annotations may differ.
# Keep separate class IDs until semantic compatibility is verified.
CLASS_NAMES = NEU_NAMES + [
    name for name in GC10_NAMES if name != "inclusion"
]
CLASS_NAMES.append("gc10_inclusion")

# NEU inclusion = 1; GC10 inclusion gets a separate ID at the end.
NEU_MAP = {i: i for i in range(len(NEU_NAMES))}
GC10_MAP = {}
for i, name in enumerate(GC10_NAMES):
    GC10_MAP[i] = (
        CLASS_NAMES.index("gc10_inclusion")
        if name == "inclusion"
        else CLASS_NAMES.index(name)
    )

def locate_image_label_roots(base):
    """Find folders named images and labels, including split subfolders."""
    image_roots = [
        p for p in base.rglob("images")
        if p.is_dir()
    ]
    label_roots = [
        p for p in base.rglob("labels")
        if p.is_dir()
    ]
    if not image_roots or not label_roots:
        raise FileNotFoundError(
            f"Could not find images/labels folders under {base}"
        )
    return image_roots, label_roots

def collect_pairs(base):
    """Match labels by relative path/stem; skip images without labels."""
    image_roots, label_roots = locate_image_label_roots(base)
    label_index = {}

    for root in label_roots:
        for label in root.rglob("*.txt"):
            # Use relative path after the labels folder to avoid split confusion.
            rel = label.relative_to(root)
            label_index[(rel.parent.as_posix(), label.stem)] = label
            label_index[("", label.stem)] = label

    pairs = []
    missing = []
    used_labels = set()

    for root in image_roots:
        for image in root.rglob("*"):
            if not image.is_file() or image.suffix.lower() not in IMAGE_EXTS:
                continue

            rel = image.relative_to(root)
            key = (rel.parent.as_posix(), image.stem)
            label = label_index.get(key) or label_index.get(("", image.stem))

            if label is None:
                missing.append(str(image))
                continue

            pairs.append((image, label))
            used_labels.add(label)

    return pairs, missing

def validate_label(label_path, mapping):
    """Validate YOLO rows and remap source class IDs."""
    output_rows = []

    for line_no, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"{label_path}:{line_no}: expected 5 YOLO values, "
                f"got {len(parts)}"
            )

        old_class = int(float(parts[0]))
        if old_class not in mapping:
            raise ValueError(
                f"{label_path}:{line_no}: unknown class ID {old_class}"
            )

        coords = [float(x) for x in parts[1:]]
        if not all(0 <= x <= 1 for x in coords):
            raise ValueError(
                f"{label_path}:{line_no}: coordinates must be 0..1"
            )

        output_rows.append(
            str(mapping[old_class]) + " " +
            " ".join(f"{x:.6f}" for x in coords)
        )

    if not output_rows:
        return ""

    return "\n".join(output_rows) + "\n"

def prepare_source(pairs, mapping, prefix, out_root):
    random.shuffle(pairs)
    n = len(pairs)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    splits = {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }

    for split, split_pairs in splits.items():
        img_dir = out_root / "images" / split
        lbl_dir = out_root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for idx, (image, label) in enumerate(split_pairs):
            # Unique names prevent NEU and GC10 filename collisions.
            name = f"{prefix}_{idx:06d}"
            destination_image = img_dir / f"{name}{image.suffix.lower()}"
            destination_label = lbl_dir / f"{name}.txt"

            shutil.copy2(image, destination_image)
            destination_label.write_text(
                validate_label(label, mapping),
                encoding="utf-8"
            )

    return len(pairs)

random.seed(SEED)

neu_pairs, neu_missing = collect_pairs(NEU)
gc_pairs, gc_missing = collect_pairs(GC10)

if not neu_pairs:
    raise RuntimeError("No matching NEU image-label pairs found.")
if not gc_pairs:
    raise RuntimeError("No matching GC10 image-label pairs found.")

# Build a clean new output directory only after matching succeeds.
OUT.mkdir(parents=True)
try:
    prepare_source(neu_pairs, NEU_MAP, "neu", OUT)
    prepare_source(gc_pairs, GC10_MAP, "gc10", OUT)

    config = {
        "path": str(OUT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }

    with (OUT / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

except Exception:
    shutil.rmtree(OUT, ignore_errors=True)
    raise

print("\nCombined dataset prepared.")
print("Output:", OUT)
print("Classes:", len(CLASS_NAMES))
for idx, name in enumerate(CLASS_NAMES):
    print(f"  {idx}: {name}")
print("NEU labeled pairs:", len(neu_pairs))
print("GC10 labeled pairs:", len(gc_pairs))
print("NEU images skipped (no matching label):", len(neu_missing))
print("GC10 images skipped (no matching label):", len(gc_missing))
print("Original dataset and models were not modified.")
