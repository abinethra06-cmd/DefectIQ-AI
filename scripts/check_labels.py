
from pathlib import Path
import random
import cv2

# Locate the prepared dataset inside the shared repository
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "data" / "neu_yolo"

SPLIT = "train"
NUMBER_OF_IMAGES = 12
SEED = 42

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def main():
    image_dir = DATASET / "images" / SPLIT
    label_dir = DATASET / "labels" / SPLIT
    output_dir = DATASET / "label_check"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(
            f"Could not find images or labels for split '{SPLIT}' "
            f"under {DATASET}"
        )

    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    random.Random(SEED).shuffle(images)
    selected = images[:min(NUMBER_OF_IMAGES, len(images))]

    for image_path in selected:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"WARNING: Could not read image: {image_path.name}")
            continue

        height, width = image.shape[:2]
        label_path = label_dir / f"{image_path.stem}.txt"
        object_count = 0

        if not label_path.exists():
            print(f"WARNING: Missing label file for {image_path.name}")
            continue

        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            parts = line.split()
            if len(parts) != 5:
                print(f"WARNING: Invalid label line in {label_path.name}:{line_number}")
                continue

            try:
                class_id = int(parts[0])
                x_center, y_center, box_width, box_height = map(float, parts[1:])
            except ValueError:
                print(f"WARNING: Non-numeric label in {label_path.name}:{line_number}")
                continue

            if not 0 <= class_id < len(CLASSES):
                print(f"WARNING: Invalid class ID in {label_path.name}:{line_number}")
                continue

            values = (x_center, y_center, box_width, box_height)
            if not all(0 <= value <= 1 for value in values):
                print(f"WARNING: Out-of-range coordinates in {label_path.name}:{line_number}")
                continue

            if box_width <= 0 or box_height <= 0:
                print(f"WARNING: Empty box in {label_path.name}:{line_number}")
                continue

            x1 = int((x_center - box_width / 2) * width)
            y1 = int((y_center - box_height / 2) * height)
            x2 = int((x_center + box_width / 2) * width)
            y2 = int((y_center + box_height / 2) * height)

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            if x2 <= x1 or y2 <= y1:
                print(f"WARNING: Invalid box in {label_path.name}:{line_number}")
                continue

            label = CLASSES[class_id]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            object_count += 1

        destination = output_dir / f"{image_path.stem}_checked.jpg"
        if not cv2.imwrite(str(destination), image):
            print(f"WARNING: Could not save {destination.name}")
            continue

        print(
            f"Saved {destination.name} | "
            f"boxes drawn: {object_count}"
        )

    print(f"\nFinished. Review the images in: {output_dir}")


if __name__ == "__main__":
    main()
