Awesome—your current setup (YOLOv8n detector + single-frame ViT behavior classifier, run end-to-end via notebooks) is a solid foundation to build on.  Below is a precise, “drop-in” upgrade plan that converts it into the hybrid modular pipeline from the report—Detection → Tracking → Temporal Classification—fully optimized for an RTX 4080 with TensorRT. I’ve kept this actionable: folder changes, code snippets, training/eval recipes, and deployment knobs.

---

# High-impact changes at a glance (delta from your repo)

1. **Add tracking between detection and classification**

   * Integrate **OC-SORT** (primary) or **ByteTrack** to persist IDs and produce per-cow clips.

2. **Upgrade behavior head from single-frame ViT → short video model**

   * Train **X3D-S/M** (recommended) or **TimeSformer-base** on 16–32-frame crops per track.

3. **Identity-safe splits + semi-auto labeling loop**

   * Generate track IDs, split by ID (not frame/video alone), and add a human-in-the-loop review pass.

4. **RTX 4080 inference optimization**

   * Export **YOLO → TensorRT (FP16)**, and the temporal classifier **→ ONNX → TensorRT**.
   * Use **NVDEC** for GPU video decode; async pipelines to saturate Tensor Cores.

5. **Production metrics**

   * Add **MOTA/IDF1** for tracking, **Top-1/F1** for behavior, and **end-to-end latency (P95)**.

---

# Repository structure (proposed)

```
cow-behavior-analysis/
├── 01_bbox_crops.ipynb
├── 02_yolo_oneclass_from_via.ipynb
├── 03_tracking_ocsort_bytetrack.ipynb      # NEW: tuning tracker, saving MOT-format tracks
├── 04_clip_maker.py                         # NEW: extract per-ID sliding-window clips
├── 05_x3d_train.py                          # NEW: train X3D on clips (temporal)
├── 06_pipeline_realtime.py                  # NEW: end-to-end (decode→det→track→classify) w/ TRT
├── 07_eval_end2end.py                       # NEW: MOTA/IDF1 + behavior metrics
├── models/
│   ├── yolo_trt/                            # TensorRT engine(s)
│   └── x3d_trt/
├── workdir/
│   ├── mot_dets/                            # per-frame detections
│   ├── mot_tracks/                          # MOT-format tracks (OC-SORT/ByteTrack)
│   └── clips/                               # (id, t0)→(T frames) crops for X3D
└── configs/
    ├── tracker.yaml                         # OC-SORT params
    ├── x3d.yaml                             # frames=16|32, stride, size, aug
    └── export.yaml                          # TRT settings
```

---

# 1) Tracking integration (OC-SORT / ByteTrack)

**Why:** Persist IDs through occlusion, build behavior clips, and enable identity-safe splits.

### Minimal integration code (Ultralytics → OC-SORT)

```python
# 03_tracking_ocsort_bytetrack.ipynb (core cell)
!pip install ultralytics opencv-python omegaconf lap==0.4.0 filterpy onemetric

import cv2, torch, numpy as np
from ultralytics import YOLO
from omegaconf import OmegaConf
from pathlib import Path
from onemetric.cv.utils.iou import box_iou_batch

# OC-SORT (reference lightweight impl)
from ocsort.ocsort import OCSort  # if using a packaged impl; otherwise vendor-in

conf = OmegaConf.load("configs/tracker.yaml")
model = YOLO("runs/detect/train/weights/best.pt")  # your trained YOLO

cap = cv2.VideoCapture("data/videos/videos/some_video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

tracker = OCSort(det_thresh=conf.det_thresh,
                 max_age=conf.max_age,
                 min_hits=conf.min_hits,
                 iou_threshold=conf.iou_thresh)

out_txt = Path("workdir/mot_tracks/some_video.txt")
out_txt.parent.mkdir(parents=True, exist_ok=True)
f = out_txt.open("w")

frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    # batched GPU det
    res = model.predict(frame, imgsz=conf.imgsz, half=True, verbose=False)[0]
    if res.boxes is None or res.boxes.shape[0] == 0:
        dets = np.empty((0, 5))
    else:
        b = res.boxes
        xyxy = b.xyxy.detach().cpu().numpy()
        confs = b.conf.detach().cpu().numpy()
        dets = np.concatenate([xyxy, confs[:, None]], axis=1)

    tracks = tracker.update(dets, frame.shape[:2], frame_idx)  # Nx6: x1,y1,x2,y2,track_id,score
    for x1,y1,x2,y2,tid,score in tracks:
        # Save MOT challenge format: frame, id, bbox(x,y,w,h), score, class,-1,-1
        w,h = x2 - x1, y2 - y1
        f.write(f"{frame_idx+1},{int(tid)},{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},{score:.3f},0,-1,-1\n")

    frame_idx += 1

f.close(); cap.release()
```

**`configs/tracker.yaml` (good defaults):**

```yaml
det_thresh: 0.2
max_age: 30
min_hits: 3
iou_thresh: 0.3
imgsz: 640
```

> Swap to **ByteTrack** by replacing the tracker calls with its `Tracker.update()`—the surrounding code stays the same. OC-SORT = speed + occlusion robustness; ByteTrack = excellent general baseline.

---

# 2) Clip extraction for temporal classifier

**Why:** Temporal models need **T-frame** sequences per ID (e.g., T=16 @ stride=2).

```python
# 04_clip_maker.py
import cv2, json
import numpy as np
from pathlib import Path
from collections import defaultdict

VIDEO = "data/videos/videos/some_video.mp4"
TRACKS = "workdir/mot_tracks/some_video.txt"
CLIPS_DIR = Path("workdir/clips"); CLIPS_DIR.mkdir(parents=True, exist_ok=True)

T = 16           # frames per clip
STRIDE = 2       # temporal stride
PAD = 0.08       # 8% padding around bbox
OUT_SIZE = 224   # crop size for X3D-S

# Load tracks into per-id dict: id -> {frame_idx: [x,y,w,h]}
tracks = defaultdict(dict)
for line in Path(TRACKS).read_text().strip().splitlines():
    f,id_,x,y,w,h,score,_,_,_ = line.split(",")
    f,id_ = int(f), int(id_)
    x,y,w,h = map(float, (x,y,w,h))
    tracks[id_][f] = (x,y,w,h)

cap = cv2.VideoCapture(VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def crop(frame, box):
    x,y,w,h = box
    # pad
    cx, cy = x + w/2, y + h/2
    w2, h2 = w*(1+PAD), h*(1+PAD)
    x1, y1 = int(max(0, cx - w2/2)), int(max(0, cy - h2/2))
    x2, y2 = int(min(width, cx + w2/2)), int(min(height, cy + h2/2))
    roi = frame[y1:y2, x1:x2]
    return cv2.resize(roi, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)

# For each id, slide a T×STRIDE window over frames where the id exists
for tid, frames in tracks.items():
    f_idxs = sorted(frames.keys())
    for i in range(0, len(f_idxs) - (T-1)*STRIDE):
        seq_idx = [f_idxs[i + k*STRIDE] for k in range(T)]
        # verify contiguous enough
        if seq_idx[-1] - seq_idx[0] > STRIDE*(T-1) + 5: 
            continue
        # read each needed frame once
        crops = []
        for fno in seq_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno-1)
            ok, frame = cap.read()
            if not ok: break
            crops.append(crop(frame, frames[fno]))
        if len(crops) == T:
            out_dir = CLIPS_DIR / f"id_{tid}"
            out_dir.mkdir(parents=True, exist_ok=True)
            # Save as npy (T,H,W,C) for fast IO
            arr = np.stack(crops, axis=0)[:, :, :, ::-1]  # BGR→RGB
            np.save(out_dir / f"{seq_idx[0]:06d}.npy", arr)

cap.release()
```

**Labeling behavior for clips**

* Use your existing VIA CSV: map each frame’s behavior to the track, then **majority-vote** (or priority order) across the T frames to assign a clip label.
* Store `clips_meta.jsonl`: `{ "path": ".../id_12/000420.npy", "label": "rumination" }`.

---

# 3) Train an X3D temporal classifier (recommended)

Leverage **torchvision**’s pretrained **X3D-S/M** (lightweight, fast, accurate). This is simpler than wiring TimeSformer at first, and it’s perfect for a 4080.

```python
# 05_x3d_train.py
!pip install torch torchvision torchaudio lightning==2.4.0

import json, torch, random
import numpy as np
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import x3d_s, X3D_XS_Weights, x3d_m
import torch.nn.functional as F

LABELS = ["stand", "rumination", "foraging", "lying_down", "drinking_water"]
L2I = {l:i for i,l in enumerate(LABELS)}

class ClipSet(Dataset):
    def __init__(self, meta_file, split):
        items = [json.loads(l) for l in Path(meta_file).read_text().splitlines()]
        # identity-safe split: meta must include "track_id"
        items = [x for x in items if x["split"] == split]
        self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]
        clip = np.load(it["path"])  # (T,224,224,3) uint8 RGB
        clip = torch.from_numpy(clip).permute(0,3,1,2).float() / 255.0  # (T,C,H,W)
        # (C,T,H,W) for torchvision video models
        clip = clip.permute(1,0,2,3)
        y = torch.tensor(L2I[it["label"]], dtype=torch.long)
        return clip, y

def make_loaders(meta):
    return (
        DataLoader(ClipSet(meta, "train"), batch_size=16, shuffle=True, num_workers=8, pin_memory=True),
        DataLoader(ClipSet(meta, "val"),   batch_size=32, shuffle=False, num_workers=8, pin_memory=True),
        DataLoader(ClipSet(meta, "test"),  batch_size=32, shuffle=False, num_workers=8, pin_memory=True),
    )

def build_model(num_classes=len(LABELS), variant="s"):
    if variant == "s":
        m = x3d_s(weights="KINETICS400_V1")
    else:
        m = x3d_m(weights="KINETICS400_V1")
    m.classifier[5] = nn.Linear(m.classifier[5].in_features, num_classes)
    return m

device = "cuda"
train_loader, val_loader, test_loader = make_loaders("workdir/clips_meta.jsonl")
model = build_model("s").to(device)
opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler()

best_val = 0.0
for epoch in range(20):
    model.train()
    for x,y in train_loader:
        x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
    # val
    model.eval(); correct=n=0
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item(); n += y.numel()
    acc = correct/n
    print(f"epoch {epoch}: val acc {acc:.4f}")
    if acc>best_val:
        best_val=acc
        torch.save(model.state_dict(), "models/x3d_s_best.pt")

# test pass omitted for brevity
```

**Notes**

* Start with **X3D-S** (fast) then consider **X3D-M** if you need more accuracy.
* Temporal aug: random temporal crop of length **T**, random horizontal flip, light color jitter.

---

# 4) Identity-safe data split + semi-auto labeling loop

* **Generate IDs with the tracker**, then split **by (video\_id, track\_id)** so any given cow’s track never leaks into validation/test.
* Use your existing YOLO to auto-propagate boxes; annotators only correct errors and assign behaviors (priority order: drinking > foraging > rumination > lying > stand, which you already use).&#x20;
* Periodically **active-learn**: sort clips by **lowest max-probability** or **highest entropy**, review those first, retrain.

---

# 5) 4080 deployment: TensorRT + NVDEC (measurable FPS gains)

### YOLO → TensorRT (FP16)

```python
# Export detector to TensorRT (Ultralytics makes this easy)
from ultralytics import YOLO
m = YOLO("runs/detect/train/weights/best.pt")
m.export(format="engine", half=True, workspace=4, imgsz=640)  # creates .engine
```

### X3D → ONNX → TensorRT

```bash
# Export to ONNX (add a tiny export script that calls torch.onnx.export with (C,T,H,W) dummy)
python export_x3d_to_onnx.py --weights models/x3d_s_best.pt --out models/x3d_s.onnx

# Build TRT engine (FP16)
trtexec --onnx=models/x3d_s.onnx --saveEngine=models/x3d_s_fp16.engine --fp16 --workspace=4096
```

### Real-time pipeline w/ GPU decode (NVDEC)

* Use **PyAV** or **FFmpeg** with `-hwaccel cuda -hwaccel_output_format cuda`, or OpenCV CUDA build, to avoid CPU bottlenecks.
* Create two async stages: **Decode+Resize → Detect(TRT) → Track(CPU) → Gather Clips → Classify(TRT)** with bounded queues.

---

# 6) End-to-end inference scaffold

```python
# 06_pipeline_realtime.py (sketch)
# - NVDEC decode frames
# - YOLO TRT inference (batched)
# - OC-SORT update per frame → track dict
# - For each track, maintain a ring buffer of crops; when len==T, classify via X3D TRT

# Pseudocode only (omitting TRT boilerplate for brevity)
from collections import deque, defaultdict

T, STRIDE = 16, 2
buffers = defaultdict(lambda: deque(maxlen=T*STRIDE))
last_class = {}

def on_frame(frame, det_boxes):
    tracks = tracker.update(det_boxes, frame.shape[:2], frame_idx)
    for x1,y1,x2,y2,tid,score in tracks:
        crop = crop_and_resize(frame, (x1,y1,x2-x1,y2-y1))
        buffers[tid].append(crop)
        if len(buffers[tid]) >= T*STRIDE:
            seq = list(buffers[tid])[::STRIDE][:T]   # T crops
            # stack → (C,T,H,W), run TRT X3D engine → logits
            label = LABELS[logits.argmax()]
            last_class[tid] = label
```

---

# 7) Evaluation you can trust

Add `07_eval_end2end.py` to compute:

* **Tracking:** MOTA, IDF1, IDs, FN/FP (use `py-motmetrics`).
* **Behavior:** Top-1 accuracy, weighted F1; **per-class** report (drinking is usually sparse).
* **End-to-End Latency:** moving average & P95 (decode→render).
* Log a small **error book**: (video, tid, t0) for the most confusing clips (entropy > τ).

---

# 8) Model & data knobs (default starting points)

* **Detector:** keep **YOLOv8n/s**; upgrade later to livestock-tuned variants if needed.
* **Tracker:** OC-SORT with `det_thresh=0.2, iou=0.3, max_age=30`.
* **Temporal model:** **X3D-S**, `T=16`, `224²`, stride=2; batch=16 (fp16).
* **Augment:** random horizontal flip, ±10% brightness/contrast; light Gaussian noise.
* **Imbalance:** oversample minority (drinking) 2–3× or focal loss (γ=2).
* **Splits:** 70/15/15 by (video\_id, track\_id).
* **Export:** FP16 first; experiment with **INT8** (calibrated) only if accuracy drop <\~1%.

---

# What you’ll get after this pass

* **Higher accuracy on dynamic behaviors** (temporal context fixes most ViT single-frame misses).
* **Stabler IDs** → better behavior trajectories and transitions.
* **Huge throughput** on a 4080 (YOLO TRT FP16 + NVDEC + small X3D), easily multi-stream.
* **Clean upgrade path** to TimeSformer or Re-ID-enhanced BoT-SORT if/when you need it.

If you want, I can turn these snippets into committed files in your layout (configs + scripts + notebooks) and tailor the export commands to your exact CUDA/cuDNN stack.
