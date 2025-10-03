# %% [markdown]
# # Cow Detection + Behavior Classification Pipeline
# 
# **Objective**: Combine YOLO detection with ViT classification for end-to-end cow behavior analysis on raw video frames.
# 
# **Dataset**: 25K labeled cow crops across 5 behaviors (drinking, foraging, lying, rumination, standing)
# 
# **Models**: 
# - YOLO cow detector (trained on bounding box annotations)
# - ViT behavior classifier (92.6% accuracy on crops)
# 
# **Pipeline**: Raw Frame → Detection → Crop Extraction → Classification → Annotated Results

# %%
# Core Python & data handling
import cv2
import numpy as np
from pathlib import Path
import subprocess

# ML libraries  
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

# Visualization
import matplotlib.pyplot as plt
from IPython.display import Video, display

# Set device and seed
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", torch.cuda.get_device_name())
torch.manual_seed(42)

# %%
# Configuration
YOLO_MODEL_PATH = "runs/detect/train3/weights/best.pt"
VIT_MODEL_PATH = "models/cow-behavior-vit"
DETECTION_CONF = 0.25
IMG_SIZE = 640
SEED = 42

# %%
# Load models with validation
assert Path(YOLO_MODEL_PATH).exists(), f"YOLO model not found: {YOLO_MODEL_PATH}"
assert Path(VIT_MODEL_PATH).exists(), f"ViT model not found: {VIT_MODEL_PATH}"

detector = YOLO(YOLO_MODEL_PATH)
processor = AutoImageProcessor.from_pretrained(VIT_MODEL_PATH, use_fast=True)
classifier = AutoModelForImageClassification.from_pretrained(VIT_MODEL_PATH).to(device)
classifier.eval()

# Get class mappings
id2label = classifier.config.id2label
class_names = list(id2label.values())
print(f"Loaded models. Classes: {class_names}")

# Quick validation test
test_crop = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    _ = classifier(test_crop)
print("✓ Pipeline ready")

# %%
# Core pipeline functions with BotSort tracking
def detect_and_track(source, persist=True):
    """Run YOLO detection with BotSort tracking"""
    results = detector.track(source=source, tracker='botsort.yaml',
                            conf=DETECTION_CONF, persist=persist, verbose=False)[0]
    if len(results.boxes) == 0:
        return [], []

    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    track_ids = results.boxes.id.cpu().numpy().astype(int) if results.boxes.id is not None else None
    return boxes, track_ids

def classify_behavior(crop):
    # Classify cow behavior from image crop
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    inputs = processor(Image.fromarray(crop_rgb), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = classifier(**inputs).logits
        probs = F.softmax(logits, dim=-1)
    
    pred_id = logits.argmax().item()
    return {"class": id2label[pred_id], "conf": probs[0][pred_id].item()}

def process_image(image_path, use_tracking=True):
    """Complete pipeline: detect + track + classify"""
    image = cv2.imread(str(image_path))
    boxes, track_ids = detect_and_track(image_path, persist=use_tracking)

    results = []
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        crop = image[y1:y2, x1:x2]
        behavior = classify_behavior(crop)
        results.append({
            "bbox": box.tolist(),
            "track_id": int(track_ids[idx]) if track_ids is not None else None,
            "behavior": behavior["class"],
            "conf": behavior["conf"]
        })

    return results

def process_video(video_path, output_path, max_frames=30):
    """Process video with BotSort tracking + behavior classification + timeline analysis"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"frames": 0, "detections": 0, "unique_tracks": 0}

    fps, width, height = (int(cap.get(p)) for p in [cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT])
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    stats = {"frames": 0, "detections": 0, "unique_tracks": set(), "track_timelines": {}, "fps": fps}

    try:
        while cap.isOpened() and stats["frames"] < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            boxes, track_ids = detect_and_track(frame, persist=True)

            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                track_id = int(track_ids[idx]) if track_ids is not None else None
                behavior = classify_behavior(crop)

                # Track timeline: record behavior per frame
                if track_id is not None:
                    stats["unique_tracks"].add(track_id)
                    stats["track_timelines"].setdefault(track_id, []).append(behavior["class"])
                    color = tuple(int(c) for c in np.random.RandomState(track_id).randint(100, 255, 3))
                else:
                    color = (0, 255, 0)

                # Draw annotations
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{track_id} | {behavior['class']} ({behavior['conf']:.2f})" if track_id else f"{behavior['class']}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            writer.write(frame)
            stats["detections"] += len(boxes)
            stats["frames"] += 1

        cap.release()
        writer.release()

        # Compute behavior durations (using actual video FPS for accurate timing)
        from collections import Counter
        stats["unique_tracks"] = len(stats["unique_tracks"])
        stats["behavior_durations"] = {}

        for tid, timeline in stats["track_timelines"].items():
            behavior_counts = Counter(timeline)
            total_frames = len(timeline)
            stats["behavior_durations"][tid] = {
                behavior: {
                    "frames": count,
                    "seconds": round(count / fps, 1),  # Accurate timing using video FPS
                    "percent": round(100 * count / total_frames, 1)
                }
                for behavior, count in behavior_counts.items()
            }

        print(f"✓ Saved: {output_path}")
        return stats

    except Exception as e:
        print(f"Error: {e}")
        cap.release()
        writer.release()
        return stats

# %%
# Visualization with BotSort tracking
def show_results(image_path, results=None, show_tracks=True):
    """Display detection + tracking results with color-coded track IDs"""
    image = cv2.imread(str(image_path))
    if results is None:
        results = process_image(image_path, use_tracking=show_tracks)

    for result in results:
        x1, y1, x2, y2 = result["bbox"]
        track_id = result.get("track_id")
        behavior, conf = result["behavior"], result["conf"]

        # Color by track ID
        color = tuple(int(c) for c in np.random.RandomState(track_id).randint(100, 255, 3)) if track_id else (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{track_id} | {behavior} ({conf:.2f})" if track_id and show_tracks else f"{behavior} ({conf:.2f})"
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    title = f"Tracked {len(results)} cows" if show_tracks else f"Detected {len(results)} cows"
    plt.title(title)
    plt.show()

    return results

# %% [markdown]
# ## Demo: Detection Only
#
# Test detection + classification without tracking:

# %%
# Test on validation images (no tracking)
val_dir = Path("workdir/yolo_cow_oneclass/images/val")
if val_dir.exists():
    test_images = list(val_dir.glob("*.jpg"))[:2]
    if test_images:
        for img_path in test_images:
            results = show_results(str(img_path), show_tracks=False)
            print(f"{img_path.name}: {len(results)} cows")
            for i, r in enumerate(results, 1):
                print(f"  {i}. {r['behavior']} ({r['conf']:.3f})")
            print()
else:
    print(f"Directory not found: {val_dir}")

# %% [markdown]
# ## Demo: BotSort Tracking
#
# Test with BotSort tracking to assign consistent IDs:

# %%
# Test with BotSort tracking
if val_dir.exists():
    test_images = list(val_dir.glob("*.jpg"))[:2]
    if test_images:
        for img_path in test_images:
            results = show_results(str(img_path), show_tracks=True)
            print(f"{img_path.name}: {len(results)} cows tracked")
            for r in results:
                tid = r.get('track_id', 'N/A')
                print(f"  Track {tid}: {r['behavior']} ({r['conf']:.3f})")
            print()
else:
    print(f"No validation images. Try: show_results('path/to/image.jpg')")

# %% [markdown]
# ## Video Processing with BotSort Tracking
#
# Process video with persistent tracking across frames:

# %%
# Video processing with BotSort tracking + behavior timeline analysis
import random
import os

video_dir = Path("data/videos/videos")
if video_dir.exists():
    video_files = list(video_dir.glob("*.mp4"))
    if video_files:
        random_video = random.choice(video_files)
        print(f"Processing: {random_video.name}")

        output_dir = Path("output_videos")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"tracked_{random_video.stem}.mp4"

        stats = process_video(str(random_video), str(output_path), max_frames=50)

        print(f"\n📊 Summary:")
        print(f"  Frames: {stats['frames']} ({stats['frames']/stats['fps']:.1f}s @ {stats['fps']} fps)")
        print(f"  Detections: {stats['detections']}")
        print(f"  Unique cows tracked: {stats['unique_tracks']}")

        print(f"\n🐄 Behavior Timeline per Cow:")
        for tid in sorted(stats.get('behavior_durations', {}).keys()):
            durations = stats['behavior_durations'][tid]
            total_time = sum(d['seconds'] for d in durations.values())
            print(f"\n  Track {tid} (tracked for {total_time}s):")

            # Sort by duration descending
            for behavior, info in sorted(durations.items(), key=lambda x: x[1]['seconds'], reverse=True):
                print(f"    • {behavior:15s}: {info['seconds']:4.1f}s ({info['percent']:4.1f}%)")

        if output_path.exists():
            print(f"\n🎬 Annotated video:")
            display(Video(str(output_path), width=640, height=480))
else:
    print(f"No videos found. Place videos in {video_dir}")

# %%



