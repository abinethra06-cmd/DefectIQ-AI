import cv2
from pathlib import Path
from src.detector import detect

# Change this to the path of your test image
image_path = Path("data/neu_yolo/images/test/scratches_160.jpg")
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Cannot read image: {image_path}")

defects, annotated_image, inference_ms = detect(
    image,
    min_conf=0.25
)

print("\n--- DEFECT DETECTION RESULTS ---")
print(f"Image: {image_path.name}")
print(f"Inference time: {inference_ms} ms")
print(f"Number of defects: {len(defects)}")

for i, defect in enumerate(defects, start=1):
    x1, y1, x2, y2 = defect["box"]

    width_px = x2 - x1
    height_px = y2 - y1

    print(f"\nDefect {i}")
    print(f"Type: {defect['type']}")
    print(f"Confidence: {defect['confidence']:.3f}")
    print(f"Width: {width_px} pixels")
    print(f"Height: {height_px} pixels")

output_path = "defect_dimensions_output.jpg"
cv2.imwrite(output_path, annotated_image)

print(f"\nAnnotated image saved to: {output_path}")

cv2.imshow("Defect Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()