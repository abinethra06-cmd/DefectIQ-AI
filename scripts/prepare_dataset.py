
from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET

# Original dataset on your laptop
SOURCE = Path(
    r"C:\Users\Rithvikha G.S\Downloads\PS19_Defect_Inspection\data\data\NEU-DET"
)

# Converted dataset will be created inside the repository's data folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "data" / "neu_yolo"

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def read_annotation(xml_path):
    """Convert one Pascal VOC XML annotation to YOLO label lines."""
    root = ET.parse(xml_path).getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing image size in {xml_path.name}")

    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions in {xml_path.name}")

    lines = []

    for obj in root.findall("object"):
        class_name = obj.findtext("name", "").strip()

        if class_name not in CLASS_TO_ID:
            raise ValueError(
                f"Unknown class '{class_name}' in {xml_path.name}"
            )

        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Missing bounding box in {xml_path.name}")

        xmin = float(box.findtext("xmin", "0"))
        ymin = float(box.findtext("ymin", "0"))
        xmax = float(box.findtext("xmax", "0"))
        ymax = float(box.findtext("ymax", "0"))

        # Clamp coordinates to image boundaries
        xmin = max(0.0, min(xmin, width))
        xmax = max(0.0, min(xmax, width))
        ymin = max(0.0, min(ymin, height))
        ymax = max(0.0, min(ymax, height))

        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"Invalid bounding box in {xml_path.name}")

        # Pascal VOC (xmin, ymin, xmax, ymax) -> YOLO normalized format
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        class_id = CLASS_TO_ID[class_name]
        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    return lines


def main():
    images_dir = SOURCE / "IMAGES"
    annotations_dir = SOURCE / "ANNOTATIONS"

    if not images_dir.is_dir() or not annotations_dir.is_dir():
        raise FileNotFoundError(
            "Could not find IMAGES and ANNOTATIONS. "
            f"Please check the SOURCE path: {SOURCE}"
        )

    # Match images and XML files by filename, ignoring extension case.
    images_by_stem = {
        p.stem.lower(): p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }

    xml_by_stem = {
        p.stem.lower(): p
        for p in annotations_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".xml"
    }

    matched_stems = sorted(set(images_by_stem) & set(xml_by_stem))
    missing_xml = sorted(set(images_by_stem) - set(xml_by_stem))
    missing_images = sorted(set(xml_by_stem) - set(images_by_stem))

    if not matched_stems:
        raise RuntimeError(
            "No matching image/XML filenames found. "
            "Check the contents of IMAGES and ANNOTATIONS."
        )

    # Validate annotations before creating output.
    records = []
    class_counts = {name: 0 for name in CLASSES}

    for stem in matched_stems:
        image_path = images_by_stem[stem]
        xml_path = xml_by_stem[stem]
        label_lines = read_annotation(xml_path)

        for line in label_lines:
            class_id = int(line.split()[0])
            class_counts[CLASSES[class_id]] += 1

        records.append((stem, image_path, label_lines))

    # Deterministic random split.
    random.Random(SEED).shuffle(records)

    total = len(records)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }

    # Refuse to overwrite an existing prepared dataset.
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(
            f"Output already contains files: {OUTPUT}\n"
            "Rename or back up that folder before running again."
        )

    for split_name, split_records in splits.items():
        image_out = OUTPUT / "images" / split_name
        label_out = OUTPUT / "labels" / split_name
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        for stem, image_path, label_lines in split_records:
            shutil.copy2(image_path, image_out / image_path.name)
            label_path = label_out / f"{stem}.txt"
            label_path.write_text(
                "\n".join(label_lines) + ("\n" if label_lines else ""),
                encoding="utf-8",
            )

    yaml_lines = [
        f"path: {OUTPUT.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(CLASSES)}",
        "names:",
    ]
    yaml_lines.extend(
        f"  {index}: {name}" for index, name in enumerate(CLASSES)
    )

    (OUTPUT / "data.yaml").write_text(
        "\n".join(yaml_lines) + "\n", encoding="utf-8"
    )

    print("\nDataset preparation completed.")
    print(f"Source: {SOURCE}")
    print(f"Output: {OUTPUT.resolve()}")
    print(f"Matched image/XML pairs: {total}")
    print(f"Images without XML: {len(missing_xml)}")
    print(f"XML files without images: {len(missing_images)}")

    for split_name, split_records in splits.items():
        print(f"{split_name}: {len(split_records)} images")

    print("\nObjects per class across the full dataset:")
    for name, count in class_counts.items():
        print(f"  {name}: {count}")

    print("\nNext: visually inspect labels before training.")


if __name__ == "__main__":
    main()
