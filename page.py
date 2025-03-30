import cv2
import numpy as np
import pyautogui

# === Screen / canvas size ===
SCREEN_W, SCREEN_H = pyautogui.size()
 # Adjust as needed to match your screen or calibration

# === Load product images ===
perfume_img = cv2.imread("/Users/alexandrakhreiche/Desktop/CV_Grp/eyetracker/perfume.png", cv2.IMREAD_UNCHANGED)
shoe_img = cv2.imread("/Users/alexandrakhreiche/Desktop/CV_Grp/eyetracker/shoes.png", cv2.IMREAD_UNCHANGED)
watch_img = cv2.imread("/Users/alexandrakhreiche/Desktop/CV_Grp/eyetracker/watch.png", cv2.IMREAD_UNCHANGED)
sunglasses_img = cv2.imread("/Users/alexandrakhreiche/Desktop/CV_Grp/eyetracker/sunglasess.png", cv2.IMREAD_UNCHANGED)

if any(img is None for img in [perfume_img, shoe_img, watch_img, sunglasses_img]):
    print("❌ Error: One or more images failed to load. Check file names.")
    exit()

# === Resize images if needed ===
def resize_img(img, max_width, max_height):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h)
    return cv2.resize(img, (int(w * scale), int(h * scale)))

perfume_img = resize_img(perfume_img, 300, 300)
shoe_img = resize_img(shoe_img, 300, 300)
watch_img = resize_img(watch_img, 300, 300)
sunglasses_img = resize_img(sunglasses_img, 300, 300)

# === Define where to place each image ===
product_positions = {
    "perfume": (100, 100),
    "shoe": (800, 100),
    "watch": (100, 600),
    "sunglasses": (800, 600)
}

product_images = {
    "perfume": perfume_img,
    "shoe": shoe_img,
    "watch": watch_img,
    "sunglasses": sunglasses_img
}

# === Create white canvas ===
canvas = np.ones((SCREEN_H, SCREEN_W, 3), dtype=np.uint8) * 255

# === Store bounding boxes ===
products = {}

# === Paste images onto canvas ===
for name, img in product_images.items():
    x, y = product_positions[name]
    h, w = img.shape[:2]

    # Handle transparency (if image has alpha channel)
    if img.shape[2] == 4:
        alpha = img[:, :, 3] / 255.0
        for c in range(3):
            canvas[y:y+h, x:x+w, c] = canvas[y:y+h, x:x+w, c] * (1 - alpha) + img[:, :, c] * alpha
    else:
        canvas[y:y+h, x:x+w] = img

    products[name] = ((x, y), (x + w, y + h))

    # Optional: label each item
    cv2.putText(canvas, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# === Show and save result ===
cv2.imshow("Product Layout", canvas)
cv2.imwrite("product_page.png", canvas)  # Save image for use in tracking
cv2.waitKey(0)
cv2.destroyAllWindows()

# === Print bounding boxes ===
print("\nProduct bounding boxes:")
for name, box in products.items():
    print(f'"{name}": {box},')
