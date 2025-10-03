#!/usr/bin/env python3
"""
Cow Behavior Analysis Pipeline with BotSort Tracking
Combines YOLO11 detection, BotSort tracking, and ViT behavior classification
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
from collections import defaultdict

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

class CowTrackingPipeline:
    """
    End-to-end pipeline for cow detection, tracking, and behavior classification
    """

    def __init__(
        self,
        yolo_model_path="yolo11n.pt",
        vit_model_path="models/cow-behavior-vit",
        detection_conf=0.25,
        tracker_config="botsort.yaml",
        device=None
    ):
        """
        Initialize the tracking pipeline

        Args:
            yolo_model_path: Path to YOLO model (supports yolo11n.pt or custom trained)
            vit_model_path: Path to ViT behavior classifier
            detection_conf: Detection confidence threshold
            tracker_config: Tracker configuration (botsort.yaml or bytetrack.yaml)
            device: Device to run on (cuda/cpu/mps, None for auto)
        """
        self.detection_conf = detection_conf
        self.tracker_config = tracker_config

        # Setup device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Initializing pipeline on {self.device}...")

        # Load YOLO detector with tracking
        print(f"Loading YOLO model: {yolo_model_path}")
        self.detector = YOLO(yolo_model_path)

        # Load ViT classifier if available
        self.classifier = None
        self.processor = None
        if Path(vit_model_path).exists():
            try:
                print(f"Loading ViT classifier: {vit_model_path}")
                self.processor = AutoImageProcessor.from_pretrained(vit_model_path, use_fast=True)
                self.classifier = AutoModelForImageClassification.from_pretrained(vit_model_path).to(self.device)
                self.classifier.eval()
                self.id2label = self.classifier.config.id2label
                print(f"✓ Classifier loaded: {list(self.id2label.values())}")
            except Exception as e:
                print(f"⚠ Could not load classifier: {e}")
                print(f"⚠ Running detection and tracking only")
        else:
            print(f"⚠ Classifier not found at {vit_model_path}, running detection and tracking only")

        print("✓ Pipeline ready")

    def detect_and_track(self, source, persist=True, verbose=False):
        """
        Run detection with BotSort tracking

        Args:
            source: Image path, video path, or frame
            persist: Persist tracks between frames
            verbose: Show detailed output

        Returns:
            Tracking results with boxes and track IDs
        """
        results = self.detector.track(
            source=source,
            tracker=self.tracker_config,
            conf=self.detection_conf,
            persist=persist,
            verbose=verbose
        )
        return results

    def classify_behavior(self, crop):
        """
        Classify cow behavior from image crop

        Args:
            crop: BGR image crop

        Returns:
            Dictionary with behavior class and confidence
        """
        if self.classifier is None:
            return {"class": "unknown", "conf": 0.0}

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        inputs = self.processor(Image.fromarray(crop_rgb), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.classifier(**inputs).logits
            probs = F.softmax(logits, dim=-1)

        pred_id = logits.argmax().item()
        return {
            "class": self.id2label[pred_id],
            "conf": probs[0][pred_id].item()
        }

    def process_frame(self, frame, persist=True):
        """
        Process a single frame with tracking and behavior classification

        Args:
            frame: BGR image frame
            persist: Persist tracks between calls

        Returns:
            List of dictionaries with tracking and behavior information
        """
        # Run detection and tracking
        results = self.detect_and_track(frame, persist=persist, verbose=False)[0]

        if len(results.boxes) == 0:
            return []

        # Extract tracking information
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results.boxes.id.cpu().numpy() if results.boxes.id is not None else None
        confidences = results.boxes.conf.cpu().numpy()

        # Process each detection
        tracked_cows = []
        for idx, (box, conf) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = box
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # Classify behavior
            behavior = self.classify_behavior(crop)

            # Compile tracking info
            cow_info = {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "track_id": int(track_ids[idx]) if track_ids is not None else None,
                "detection_conf": float(conf),
                "behavior": behavior["class"],
                "behavior_conf": behavior["conf"]
            }
            tracked_cows.append(cow_info)

        return tracked_cows

    def process_video(
        self,
        video_path,
        output_path=None,
        max_frames=None,
        show_tracks=True,
        show_behavior=True
    ):
        """
        Process video with tracking and behavior classification

        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (None = display only)
            max_frames: Maximum frames to process (None = all)
            show_tracks: Show track IDs on video
            show_behavior: Show behavior labels on video

        Returns:
            Dictionary with processing statistics and track history
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"✗ Could not open video: {video_path}")
            return None

        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\nProcessing video: {Path(video_path).name}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        if max_frames:
            print(f"  Processing: {max_frames} frames")

        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Track statistics
        stats = {
            "frames_processed": 0,
            "total_detections": 0,
            "unique_tracks": set(),
            "track_history": defaultdict(list),  # track_id -> list of behaviors
            "behavior_counts": defaultdict(int)
        }

        # Process frames
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames and frame_count >= max_frames):
                break

            # Process frame
            tracked_cows = self.process_frame(frame, persist=True)

            # Update statistics
            stats["frames_processed"] += 1
            stats["total_detections"] += len(tracked_cows)

            # Annotate frame
            for cow in tracked_cows:
                x1, y1, x2, y2 = cow["bbox"]
                track_id = cow["track_id"]
                behavior = cow["behavior"]
                behavior_conf = cow["behavior_conf"]

                # Update track history
                if track_id is not None:
                    stats["unique_tracks"].add(track_id)
                    stats["track_history"][track_id].append(behavior)

                stats["behavior_counts"][behavior] += 1

                # Draw bounding box (color by track ID if available)
                if track_id is not None:
                    # Generate color from track ID
                    color = tuple(int(c) for c in np.random.RandomState(track_id).randint(0, 255, 3))
                else:
                    color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Prepare label
                label_parts = []
                if show_tracks and track_id is not None:
                    label_parts.append(f"ID:{track_id}")
                if show_behavior:
                    label_parts.append(f"{behavior} ({behavior_conf:.2f})")

                label = " | ".join(label_parts) if label_parts else ""

                if label:
                    # Draw label background
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1-20), (x1+label_w, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Write frame
            if writer:
                writer.write(frame)

            frame_count += 1

            # Progress update
            if frame_count % 30 == 0:
                print(f"  Processed {frame_count} frames, {len(stats['unique_tracks'])} unique tracks")

        # Cleanup
        cap.release()
        if writer:
            writer.release()
            print(f"\n✓ Saved annotated video: {output_path}")

        # Finalize statistics
        stats["unique_tracks"] = len(stats["unique_tracks"])

        # Calculate dominant behavior per track
        stats["track_behaviors"] = {}
        for track_id, behaviors in stats["track_history"].items():
            if behaviors:
                stats["track_behaviors"][track_id] = max(set(behaviors), key=behaviors.count)

        return stats


def main():
    """Demo usage"""
    print("=" * 60)
    print("Cow Tracking Pipeline with BotSort")
    print("=" * 60)

    # Initialize pipeline
    pipeline = CowTrackingPipeline(
        yolo_model_path="yolo11n.pt",  # Will download if not present
        vit_model_path="models/cow-behavior-vit",
        tracker_config="botsort.yaml"
    )

    print("\n" + "=" * 60)
    print("Pipeline initialized and ready for use!")
    print("=" * 60)
    print("\nExample usage:")
    print("  # Process a video")
    print("  stats = pipeline.process_video('video.mp4', 'output.mp4')")
    print("\n  # Process a single frame")
    print("  tracked_cows = pipeline.process_frame(frame)")
    print("=" * 60)


if __name__ == "__main__":
    main()
