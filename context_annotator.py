import os
import cv2
import torch
import clip
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- SETTINGS ----------------
train_folder_path = r"C:\Users\surya\Desktop\computer vision\Hopfield networks\best_context_images_2"
obb_label_path = r"C:\Users\surya\Desktop\computer vision\Hopfield networks\best_context_labels_2"
context_label_path = r"C:\Users\surya\Desktop\computer vision\Hopfield networks\best_context_proposal_label_2"

os.makedirs(context_label_path, exist_ok=True)

# ---------------- CLIP SETUP ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

def clip_embed_image(img_np):
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_preprocessed = preprocess(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img_preprocessed)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0).cpu().numpy()

def draw_obb(img, obb_pts, offset_x=0, offset_y=0):
    vis = img.copy()
    pts = np.array([[int(x - offset_x), int(y - offset_y)] for (x, y) in obb_pts], dtype=np.int32)
    cv2.polylines(vis, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB) # Convert for Matplotlib

# ---------------- ANNOTATION LOOP ----------------
images = [f for f in os.listdir(train_folder_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

print(f"Total images found: {len(images)}")

for image_name in images:
    img_path = os.path.join(train_folder_path, image_name)
    label_path = os.path.join(obb_label_path, image_name.rsplit('.', 1)[0] + ".txt")

    if not os.path.exists(label_path):
        continue

    img = cv2.imread(img_path)
    if img is None: continue
    H, W = img.shape[:2]

    with open(label_path, "r") as f:
        lines = f.readlines()

    for obj_idx, line in enumerate(lines):
        # RESUME LOGIC: Skip if file exists
        save_name = f"{image_name.rsplit('.', 1)[0]}_obj{obj_idx}.npz"
        save_path = os.path.join(context_label_path, save_name)
        if os.path.exists(save_path):
            print(f"Skipping {save_name} (Already exists)")
            continue

        parts = line.strip().split()
        cls = int(parts[0])
        x1,y1,x2,y2,x3,y3,x4,y4 = map(float, parts[1:9])
        obb_pts = [(x1*W, y1*H), (x2*W, y2*H), (x3*W, y3*H), (x4*W, y4*H)]
        
        # Calculate crops
        xs, ys = [p[0] for p in obb_pts], [p[1] for p in obb_pts]
        xmin, xmax = max(0, int(min(xs))), min(W, int(max(xs)))
        ymin, ymax = max(0, int(min(ys))), min(H, int(max(ys)))
        
        bw, bh = xmax - xmin, ymax - ymin
        cx, cy = (xmin + xmax)//2, (ymin + ymax)//2
        exmin, exmax = max(0, cx - bw), min(W, cx + bw)
        eymin, eymax = max(0, cy - bh), min(H, cy + bh)

        # Context options
        contexts = {
            '1': img[:, xmin:xmax],
            '2': img[ymin:ymax, :],
            '3': img[eymin:eymax, exmin:exmax],
            '4': img
        }

        # Visualization Grid
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"Image: {image_name} | Obj: {obj_idx}\nPress 1, 2, 3, 4 to select | Space to skip | Q to quit", fontsize=12)
        
        axes[0].imshow(draw_obb(contexts['1'], obb_pts, xmin, 0))
        axes[0].set_title("1: Column")
        axes[1].imshow(draw_obb(contexts['2'], obb_pts, 0, ymin))
        axes[1].set_title("2: Row")
        axes[2].imshow(draw_obb(contexts['3'], obb_pts, exmin, eymin))
        axes[2].set_title("3: Box2x")
        axes[3].imshow(draw_obb(contexts['4'], obb_pts, 0, 0))
        axes[3].set_title("4: Full")
        
        for ax in axes: ax.axis('off')

        selected_key = [None]
        def on_key(event):
            if event.key in ['1', '2', '3', '4', 'q', ' ']:
                selected_key[0] = event.key
                plt.close()

        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.tight_layout()
        plt.show()

        key = selected_key[0]
        if key == 'q':
            print("Quitting...")
            exit()
        elif key in ['1', '2', '3', '4']:
            emb = clip_embed_image(contexts[key])
            np.savez(save_path, cls=cls, embedding=emb)
            print(f"Saved → {save_name}")
        else:
            print("Skipped.")

print("All done!")