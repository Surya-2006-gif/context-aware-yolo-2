import torch
import clip
import torch.nn as nn
import pickle
import torch.nn.functional as F
import os
import shutil
import numpy as np
from PIL import Image
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from ultralytics import YOLO

classes_dict = {
0:"screwdriver",
1:"comb",
2:"knife",
3:"pen",
4:"toothbrush"
}

# ------------------ ATTENTION + MHA + MLP ------------------

class attn(nn.Module):
    def __init__(self, dim_size=64):
        super().__init__()
        self.key = nn.Linear(dim_size, dim_size)
        self.query = nn.Linear(dim_size, dim_size)
        self.value = nn.Linear(dim_size, dim_size)

    def forward(self, x, y):
        K = self.key(x)
        Q = self.query(y)
        V = self.value(x)
        d_k = K.size(-1)
        attn_weights = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (d_k**0.5), dim=-1)
        return torch.matmul(attn_weights, V)

class MHA(nn.Module):
    def __init__(self, num_heads=8, dim_size=512):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_size // num_heads
        self.MHA = nn.ModuleList([attn(self.head_dim) for _ in range(num_heads)])
        self.norm = nn.LayerNorm(dim_size)
    
    def forward(self, x, y):
        attn_outs = []
        for i, head in enumerate(self.MHA):
            start = i * self.head_dim
            end = (i + 1) * self.head_dim
            x_slice = x[:, :, start:end]
            y_slice = y[:, :, start:end]
            attn_outs.append(head(x_slice, y_slice))

        out = torch.cat(attn_outs, dim=-1)
        out = self.norm(out + y)
        return out

class MLP(nn.Module):
    def __init__(self, dim_size=512, hidden_dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(dim_size, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, dim_size, bias=False)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(dim_size)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.norm(out + residual)
        return out

class ContextAggregator(nn.Module):
    def __init__(self, dim_size=512):
        super().__init__()
        self.query_projection = nn.Linear(dim_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.out_projection = nn.Sequential(
            nn.Linear(dim_size, dim_size),
            nn.LayerNorm(dim_size),
            nn.GELU()
        )

    def forward(self, x):
        attention_scores = self.query_projection(x)
        weights = self.softmax(attention_scores)      # (b,4,1)
        combined = torch.matmul(weights.transpose(-2,-1), x)
        out = self.out_projection(combined)
        return out

# ------------------ CONTEXT SELECTOR MODEL ------------------

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.MHAmoduoles = nn.ModuleList([MHA(num_heads=8, dim_size=512) for _ in range(4)])
        self.mlp = MLP()
        self.aggregator = ContextAggregator(dim_size=512)

    def forward(self, x, y):
        for mha in self.MHAmoduoles:
            result = mha(x, y)
            x = x + result
        x = self.mlp(x)
        final_prediction = self.aggregator(x)  # (b,1,512)
        return final_prediction


# ------------------ HELPERS ------------------

def clip_embed_image(img_np, clip_model, preprocess, device):
    img_pil = Image.fromarray(img_np).convert("RGB")
    img_tensor = preprocess(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = clip_model.encode_image(img_tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    return emb.squeeze(0)

def Hopfield_update(beta=8, memory_matrix=None, query_emb=None, device="cpu"):
    assert memory_matrix is not None and query_emb is not None
    
    if isinstance(memory_matrix, np.ndarray):
        X = torch.from_numpy(memory_matrix).float().to(device)
    else:
        X = memory_matrix.clone().detach().float().to(device)

    if isinstance(query_emb, np.ndarray):
        q = torch.from_numpy(query_emb).float().to(device)
    else:
        q = query_emb.clone().detach().float().to(device)

    q = q.view(-1)
    scores = beta * torch.matmul(X, q)
    weights = F.softmax(scores, dim=0)
    xi_new = torch.matmul(weights, X)
    return xi_new

def tokenize_texts(texts, clip_model, device):
    if isinstance(texts, str):
        texts = [texts]

    text_tokens = clip.tokenize(texts).to(device)

    with torch.no_grad():
        text_embeddings = clip_model.encode_text(text_tokens)

    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    return text_embeddings


# ------------------ YOLO DETECTION ------------------

class detection_module:
    def __init__(self, model):
        self.model = model

    def test(self, img):
        results = self.model(img, conf=0.25)
        result_list = []

        for r in results:
            if r.obb is None:
                continue

            obb = r.obb

            result_dict = {
                "cls": obb.cls.cpu().numpy().astype(int),
                "conf": obb.conf.cpu().numpy(),
                "obb": obb.xyxyxyxy.cpu().numpy()
            }

            result_list.append(result_dict)

        return result_list


# ------------------ HIERARCHICAL CONTEXT USING SELECTOR ------------------

class hierarchical_context:
    def __init__(self, clip_model, preprocess, device):
        self.clip_model = clip_model
        self.preprocess = preprocess
        self.device = device

        self.selector = model().to(device)
        self.selector.eval()

    def get_best_context(self, img, detections_list):

        H, W = img.shape[:2]
        assert H == 896 and W == 896

        results = {}

        # case: single object
        if len(detections_list) == 1:
            det = detections_list[0]

            full_resized = cv2.resize(img, (224, 224))
            full_emb = clip_embed_image(full_resized, self.clip_model, self.preprocess, self.device)

            results[0] = {
                "cls": det["cls"],
                "best_context_emb": full_emb,
                "best_context_type": "full",
                "score": 1.0
            }
            return results

        # multiple objects
        for idx, det in enumerate(detections_list):

            obb = det["obb"]
            cls_id = det["cls"]

            xs = [p[0] for p in obb]
            ys = [p[1] for p in obb]

            xmin, xmax = int(min(xs)), int(max(xs))
            ymin, ymax = int(min(ys)), int(max(ys))

            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(W, xmax), min(H, ymax)

            # candidate contexts
            contexts = {
                "col": img[:, xmin:xmax],
                "row": img[ymin:ymax, :],
                "full": img
            }

            bw, bh = xmax - xmin, ymax - ymin
            cx, cy = (xmin + xmax)//2, (ymin + ymax)//2

            exmin = max(0, cx - bw)
            exmax = min(W, cx + bw)
            eymin = max(0, cy - bh)
            eymax = min(H, cy + bh)

            contexts["box2x"] = img[eymin:eymax, exmin:exmax]

            # ----- embed all 4 contexts -----
            context_embs = []
            for name, ctx in contexts.items():
                ctx_resized = cv2.resize(ctx, (224, 224))
                ctx_emb = clip_embed_image(ctx_resized, self.clip_model, self.preprocess, self.device)
                context_embs.append((name, ctx_emb))

            names = [n for n, _ in context_embs]
            ctx_tensor = torch.stack([e for _, e in context_embs], dim=0)    # (4,512)
            ctx_tensor = ctx_tensor.unsqueeze(0).to(self.device)            # (1,4,512)

            with torch.no_grad():
                selector_out = self.selector(ctx_tensor, ctx_tensor)        # (1,1,512)

            selector_out = selector_out.squeeze(0).squeeze(0)

            best_idx = torch.argmax(
                torch.tensor([
                    torch.dot(selector_out, e).item()
                    for _, e in context_embs
                ])
            )

            best_name, best_emb = context_embs[best_idx]
            best_score = torch.dot(selector_out, best_emb).item()

            results[idx] = {
                "cls": cls_id,
                "best_context_emb": best_emb,
                "best_context_type": best_name,
                "score": best_score
            }

        return results


# ------------------ MAIN ------------------

if __name__ =='__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    print("Loading CLIP...")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    YOLO_WEIGHTS = r"C:\Users\surya\Desktop\computer vision\Hopfield networks\obb\train4\weights\best.pt"
    MEDIAN_EMBEDDINGS_PATH = r"C:\Users\surya\Desktop\computer vision\Hopfield networks\median_matrix.pkl"

    print("Loading context memory...")
    with open(MEDIAN_EMBEDDINGS_PATH, 'rb') as f:
        MEMORY_BANK = pickle.load(f)

    print("Loading YOLO...")
    yolo_model = YOLO(YOLO_WEIGHTS)

    image = cv2.imread(r"C:\Users\surya\Desktop\computer vision\Hopfield networks\test_2.png")
    image = cv2.resize(image,(896,896))

    detector = detection_module(yolo_model)
    results = detector.test(image)

    hier_context_module = hierarchical_context(clip_model, preprocess, device)
    best_contexts = hier_context_module.get_best_context(image, results)

    output_image = image.copy()

    for context_key, context_value in best_contexts.items():

        cls_id = int(context_value["cls"][0])
        context_emb = context_value["best_context_emb"]

        pred_label = classes_dict.get(int(cls_id),"unknown")
        pred_label_text_embedding = tokenize_texts(pred_label, clip_model, device)

        retrived_context_emb = Hopfield_update(
            memory_matrix=MEMORY_BANK,
            query_emb=context_emb,
            device=device)

        retrived_pred_emb = Hopfield_update(
            memory_matrix=MEMORY_BANK,
            query_emb=pred_label_text_embedding.squeeze(0),
            device=device)

        sim = cosine_similarity(
            retrived_context_emb.cpu().numpy().reshape(1,-1),
            retrived_pred_emb.cpu().numpy().reshape(1,-1)
        )

        raw_obb = results[context_key]["obb"]
        pts = raw_obb.astype(np.int32).reshape((-1, 1, 2))

        score_val = sim[0,0]

        if score_val > 0.9:
            status_text = f"{pred_label}: CORRECT ({score_val:.2f})"
            box_color = (0, 255, 0)
            text_color = (0, 255, 0)
        else:
            status_text = f"{pred_label}: BAD CONTEXT ({score_val:.2f})"
            box_color = (0, 0, 255)
            text_color = (0, 0, 255)

        cv2.polylines(output_image, [pts], isClosed=True, color=box_color, thickness=3)

        top_point = np.min(pts, axis=0)[0]
        text_pos = (max(0, top_point[0]), max(20, top_point[1] - 10))

        cv2.putText(output_image, status_text, text_pos, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        print(f"Processed {pred_label} | Score: {score_val:.3f}")

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Context Aware Object Verification")
    plt.show()
