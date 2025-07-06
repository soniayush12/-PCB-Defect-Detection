import cv2
import os
from ultralytics import YOLO
from tkinter import Tk, filedialog
from collections import Counter

model = YOLO("best.pt")

def choose_image_file():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title='Select PCB Image',
        filetypes=[('Image Files', '*.jpg *.png *.jpeg')]
    )
    return file_path

def draw_boxes(image, results):
    """Draw bounding boxes and confidence only."""
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)
        color = (0, 0, 255)  # Red color
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"{conf:.2f}"
        cv2.putText(image, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

def detect_defects(image_path):
    print(f"\n🔍 Analyzing image: {image_path}")
    img = cv2.imread(image_path)
    results = model(img)[0]

    if len(results.boxes) == 0:
        print("✅ No defects detected.")

    print("\n🧾 Detected Defects:")
    for box in results.boxes:
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        print(f"- Confidence: {conf:.2f} at ({x1}, {y1}) to ({x2}, {y2})")

    annotated = draw_boxes(img.copy(), results)
    cv2.imshow("🔬 Defect Detection Result", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    file_path = choose_image_file()
    if file_path and os.path.exists(file_path):
        detect_defects(file_path)
    else:
        print("❌ No valid image selected.")
