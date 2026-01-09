import os
import cv2
import random
import numpy as np

# ---------------- PATHS ----------------
train_img_path = r"C:\Users\surya\Desktop\computer vision\Hopfield networks\dataset\images\train"
train_lbl_path = r"C:\Users\surya\Desktop\computer vision\Hopfield networks\dataset\labels\train"

images_save_path = r"C:\Users\surya\Desktop\computer vision\Hopfield networks\best_context_images"
labels_save_path = r"C:\Users\surya\Desktop\computer vision\Hopfield networks\best_context_labels"

os.makedirs(images_save_path, exist_ok=True)
os.makedirs(labels_save_path, exist_ok=True)

merge_probability = 0.5

# ---------------- LOAD IMAGES ----------------
image_files = [
    f for f in os.listdir(train_img_path)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

save_idx = 0

# ---------------- HELPER: READ OBB LABELS ----------------
def read_obb_labels(label_path):
    labels = []
    if not os.path.exists(label_path):
        return labels

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            coords = list(map(float, parts[1:9]))  # normalized
            labels.append((cls, coords))
    return labels

# ---------------- HELPER: WRITE OBB LABELS ----------------
def write_obb_labels(label_path, labels):
    with open(label_path, "w") as f:
        for cls, coords in labels:
            coord_str = " ".join(f"{c:.6f}" for c in coords)
            f.write(f"{cls} {coord_str}\n")

# ---------------- MERGE FUNCTION ----------------
def merge_images_and_labels(img1, lbl1, img2, lbl2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    mode = random.choice(["horizontal", "vertical"])

    merged_labels = []

    if mode == "horizontal":
        h = min(h1, h2)

        img1 = cv2.resize(img1, (int(w1 * h / h1), h))
        img2 = cv2.resize(img2, (int(w2 * h / h2), h))

        w1n, w2n = img1.shape[1], img2.shape[1]
        merged_img = np.hstack([img1, img2])
        Hm, Wm = h, w1n + w2n

        # labels from image 1
        for cls, coords in lbl1:
            pts = np.array(coords).reshape(4, 2)
            pts[:, 0] *= w1n
            pts[:, 1] *= Hm
            pts[:, 0] /= Wm
            pts[:, 1] /= Hm
            merged_labels.append((cls, pts.flatten().tolist()))

        # labels from image 2 (shift X)
        for cls, coords in lbl2:
            pts = np.array(coords).reshape(4, 2)
            pts[:, 0] *= w2n
            pts[:, 1] *= Hm
            pts[:, 0] += w1n
            pts[:, 0] /= Wm
            pts[:, 1] /= Hm
            merged_labels.append((cls, pts.flatten().tolist()))

    else:  # vertical
        w = min(w1, w2)

        img1 = cv2.resize(img1, (w, int(h1 * w / w1)))
        img2 = cv2.resize(img2, (w, int(h2 * w / w2)))

        h1n, h2n = img1.shape[0], img2.shape[0]
        merged_img = np.vstack([img1, img2])
        Hm, Wm = h1n + h2n, w

        for cls, coords in lbl1:
            pts = np.array(coords).reshape(4, 2)
            pts[:, 0] *= Wm
            pts[:, 1] *= h1n
            pts[:, 0] /= Wm
            pts[:, 1] /= Hm
            merged_labels.append((cls, pts.flatten().tolist()))

        for cls, coords in lbl2:
            pts = np.array(coords).reshape(4, 2)
            pts[:, 0] *= Wm
            pts[:, 1] *= h2n
            pts[:, 1] += h1n
            pts[:, 0] /= Wm
            pts[:, 1] /= Hm
            merged_labels.append((cls, pts.flatten().tolist()))

    return merged_img, merged_labels, mode

# ---------------- MAIN LOOP ----------------
for img_name in image_files:

    img_path = os.path.join(train_img_path, img_name)
    lbl_path = os.path.join(train_lbl_path, img_name.replace(".jpg", ".txt"))

    img = cv2.imread(img_path)
    lbl = read_obb_labels(lbl_path)

    if img is None:
        continue

    if random.random() < merge_probability:
        partner = random.choice(image_files)
        img2 = cv2.imread(os.path.join(train_img_path, partner))
        lbl2 = read_obb_labels(os.path.join(train_lbl_path, partner.replace(".jpg", ".txt")))

        if img2 is not None:
            merged_img, merged_lbl, mode = merge_images_and_labels(img, lbl, img2, lbl2)

            out_img = f"merged_{save_idx}_{mode}.jpg"
            out_lbl = out_img.replace(".jpg", ".txt")

            cv2.imwrite(os.path.join(images_save_path, out_img), merged_img)
            write_obb_labels(os.path.join(labels_save_path, out_lbl), merged_lbl)

            save_idx += 1
            continue

    # ---- save original
    out_img = f"single_{save_idx}.jpg"
    out_lbl = out_img.replace(".jpg", ".txt")

    cv2.imwrite(os.path.join(images_save_path, out_img), img)
    write_obb_labels(os.path.join(labels_save_path, out_lbl), lbl)

    save_idx += 1

# ---------------- SUMMARY ----------------
print(f"Original images : {len(image_files)}")
print(f"Saved images    : {save_idx}")
print("Done ✅")
