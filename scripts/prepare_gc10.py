
from pathlib import Path
import yaml

# GC10-DET dataset downloaded to your Downloads folder
dataset_dir = (
    Path.home() / "Downloads" / "GC10-DET" / "data"
)

config_file = dataset_dir / "data.yaml"

if not config_file.exists():
    raise FileNotFoundError(
        f"Cannot find dataset configuration: {config_file}"
    )

with config_file.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

print("\nGC10-DET configuration")
print("----------------------")
print("Dataset folder:", dataset_dir)
print("Class count:", config.get("nc", "not specified"))
print("Class names:", config.get("names", "not specified"))

images_dir = dataset_dir / "images"
labels_dir = dataset_dir / "labels"

print("\nImage folder exists:", images_dir.exists())
print("Label folder exists:", labels_dir.exists())

if images_dir.exists():
    image_count = sum(
        1 for p in images_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    print("Images found:", image_count)

if labels_dir.exists():
    label_count = sum(1 for _ in labels_dir.rglob("*.txt"))
    print("Label files found:", label_count)

print("\nInspection complete. No files were changed.")
