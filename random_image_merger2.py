import os
import cv2
import random
import numpy as np
import math
from tqdm import tqdm # Optional: pip install tqdm

# ---------------- PATHS ----------------
train_img_path = r"C:\Users\surya\Desktop\computer vision\Hopfield networks\dataset\images\train"
train_lbl_path = r"C:\Users\surya\Desktop\computer vision\Hopfield networks\dataset\labels\train"
images_save_path = r"C:\Users\surya\Desktop\computer vision\Hopfield networks\best_context_images_2"
labels_save_path = r"C:\Users\surya\Desktop\computer vision\Hopfield networks\best_context_labels_2"

os.makedirs(images_save_path, exist_ok=True)
os.makedirs(labels_save_path, exist_ok=True)

# ---------------- HELPERS ----------------
def read_obb_labels(label_path):
    labels = []
    if not os.path.exists(label_path): return labels
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9: continue
            labels.append((int(parts[0]), list(map(float, parts[1:9]))))
    return labels

def write_obb_labels(label_path, labels):
    with open(label_path, "w") as f:
        for cls, coords in labels:
            f.write(f"{cls} " + " ".join(f"{c:.6f}" for c in coords) + "\n")

def rotate_image_and_obb(img, labels, angle_range=(-30, 30)):
    h, w = img.shape[:2]
    angle = random.uniform(*angle_range)
    center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    
    rotated_img = cv2.warpAffine(img, M, (nW, nH))
    
    new_labels = []
    for cls, coords in labels:
        pts = np.array(coords).reshape(4, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        ones = np.ones(shape=(len(pts), 1))
        pts_ones = np.hstack([pts, ones])
        transformed_pts = M.dot(pts_ones.T).T
        transformed_pts[:, 0] /= nW
        transformed_pts[:, 1] /= nH
        new_labels.append((cls, transformed_pts.flatten().tolist()))
        
    return rotated_img, new_labels

def resize_and_pad(img, labels, target_w, target_h):
    h, w = img.shape[:2]
    scale = min(target_w/w, target_h/h)
    nw, nh = int(w * scale), int(h * scale)
    
    img_resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    dx, dy = (target_w - nw) // 2, (target_h - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = img_resized
    
    new_labels = []
    for cls, coords in labels:
        pts = np.array(coords).reshape(4, 2)
        pts[:, 0] = (pts[:, 0] * nw + dx) / target_w
        pts[:, 1] = (pts[:, 1] * nh + dy) / target_h
        new_labels.append((cls, pts.flatten().tolist()))
    return canvas, new_labels

# ---------------- PREPARATION ----------------
image_files = [f for f in os.listdir(train_img_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

# --- KEY ADDITION: SHUFFLE ---
random.shuffle(image_files)
print(f"Dataset shuffled. Total images: {len(image_files)}")

save_idx = 0
pbar = tqdm(total=len(image_files))

# ---------------- PROCESSING ----------------
i = 0
while i < len(image_files):
    prob = random.random()
    
    # 1. NO MERGE (25%)
    if prob < 0.25:
        img = cv2.imread(os.path.join(train_img_path, image_files[i]))
        lbl = read_obb_labels(os.path.join(train_lbl_path, image_files[i].rsplit('.', 1)[0] + ".txt"))
        if img is not None:
            out_name = f"single_{save_idx}"
            cv2.imwrite(os.path.join(images_save_path, out_name + ".jpg"), img)
            write_obb_labels(os.path.join(labels_save_path, out_name + ".txt"), lbl)
        i += 1
        pbar.update(1)

    # 2. TILT & MERGE 2 IMAGES (25%)
    elif prob < 0.50 and i + 1 < len(image_files):
        pair = []
        valid = True
        for j in range(2):
            img = cv2.imread(os.path.join(train_img_path, image_files[i+j]))
            lbl = read_obb_labels(os.path.join(train_lbl_path, image_files[i+j].rsplit('.', 1)[0] + ".txt"))
            if img is None: 
                valid = False
                break
            pair.append(rotate_image_and_obb(img, lbl))
        
        if valid:
            h_max = max(pair[0][0].shape[0], pair[1][0].shape[0])
            w_total = pair[0][0].shape[1] + pair[1][0].shape[1]
            canvas = np.zeros((h_max, w_total, 3), dtype=np.uint8)
            
            merged_lbls = []
            curr_x = 0
            for img_rot, lbl_rot in pair:
                ch, cw = img_rot.shape[:2]
                canvas[0:ch, curr_x:curr_x+cw] = img_rot
                for cls, coords in lbl_rot:
                    pts = np.array(coords).reshape(4, 2)
                    pts[:, 0] = (pts[:, 0] * cw + curr_x) / w_total
                    pts[:, 1] = (pts[:, 1] * ch) / h_max
                    merged_lbls.append((cls, pts.flatten().tolist()))
                curr_x += cw
                
            out_name = f"tilted_{save_idx}"
            cv2.imwrite(os.path.join(images_save_path, out_name + ".jpg"), canvas)
            write_obb_labels(os.path.join(labels_save_path, out_name + ".txt"), merged_lbls)
        i += 2
        pbar.update(2)

    # 3. MERGE 4 IMAGES (50%)
    elif i + 3 < len(image_files):
        quad = []
        valid = True
        for j in range(4):
            img = cv2.imread(os.path.join(train_img_path, image_files[i+j]))
            lbl = read_obb_labels(os.path.join(train_lbl_path, image_files[i+j].rsplit('.', 1)[0] + ".txt"))
            if img is None:
                valid = False
                break
            quad.append((img, lbl))
        
        if valid:
            mode_prob = random.random()
            # 70% Square, 15% Horz, 15% Vert
            if mode_prob < 0.70: # Square (2x2)
                tw, th = 640, 640
                final_lbls = []
                rows = []
                for r in range(2):
                    cols = []
                    for c in range(2):
                        img_p, lbl_p = resize_and_pad(quad[r*2+c][0], quad[r*2+c][1], tw, th)
                        cols.append(img_p)
                        for cls, coords in lbl_p:
                            pts = np.array(coords).reshape(4, 2)
                            pts[:, 0] = (pts[:, 0] * tw + (c * tw)) / (tw * 2)
                            pts[:, 1] = (pts[:, 1] * th + (r * th)) / (th * 2)
                            final_lbls.append((cls, pts.flatten().tolist()))
                    rows.append(np.hstack(cols))
                canvas = np.vstack(rows)
            
            elif mode_prob < 0.85: # Horizontal (1x4)
                tw, th = 400, 800
                cols, final_lbls = [], []
                for c in range(4):
                    img_p, lbl_p = resize_and_pad(quad[c][0], quad[c][1], tw, th)
                    cols.append(img_p)
                    for cls, coords in lbl_p:
                        pts = np.array(coords).reshape(4, 2)
                        pts[:, 0] = (pts[:, 0] * tw + (c * tw)) / (tw * 4)
                        pts[:, 1] = pts[:, 1] 
                        final_lbls.append((cls, pts.flatten().tolist()))
                canvas = np.hstack(cols)
            else: # Vertical (4x1)
                tw, th = 800, 400
                rows, final_lbls = [], []
                for r in range(4):
                    img_p, lbl_p = resize_and_pad(quad[r][0], quad[r][1], tw, th)
                    rows.append(img_p)
                    for cls, coords in lbl_p:
                        pts = np.array(coords).reshape(4, 2)
                        pts[:, 0] = pts[:, 0]
                        pts[:, 1] = (pts[:, 1] * th + (r * th)) / (th * 4)
                        final_lbls.append((cls, pts.flatten().tolist()))
                canvas = np.vstack(rows)

            out_name = f"quad_{save_idx}"
            cv2.imwrite(os.path.join(images_save_path, out_name + ".jpg"), canvas)
            write_obb_labels(os.path.join(labels_save_path, out_name + ".txt"), final_lbls)
        i += 4
        pbar.update(4)
    else:
        i += 1
        pbar.update(1)

    save_idx += 1

pbar.close()
print(f"\nDone! Shuffled and processed {save_idx} total samples.")