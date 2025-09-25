#!/usr/bin/env python3
"""
Clip Maker for Cow Behavior Analysis

Extracts temporal clips from MOT tracking data for training temporal classifiers (X3D).
Takes MOT-format tracking files and creates T-frame sequences per cow ID.

Usage:
    python 04_clip_maker.py --video path/to/video.mp4 --tracks path/to/tracks.txt
    python 04_clip_maker.py --config configs/clip_maker.yaml --video data/video.mp4
"""

import argparse
import cv2
import json
import numpy as np
import yaml
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ClipMaker:
    def __init__(self, config_path: str = "configs/clip_maker.yaml"):
        """Initialize ClipMaker with configuration."""
        self.config = self.load_config(config_path)
        self.T = self.config['T']
        self.STRIDE = self.config['STRIDE']
        self.MAX_GAP = self.config['MAX_GAP']
        self.PAD = self.config['PAD']
        self.OUT_SIZE = self.config['OUT_SIZE']
        self.MIN_TRACK_LENGTH = self.config['MIN_TRACK_LENGTH']

        # Create output directories
        self.clips_dir = Path("workdir/clips")
        self.clips_dir.mkdir(parents=True, exist_ok=True)

    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self.default_config()

    def default_config(self) -> dict:
        """Return default configuration if config file not found."""
        return {
            'T': 16,
            'STRIDE': 2,
            'MAX_GAP': 5,
            'PAD': 0.08,
            'OUT_SIZE': 224,
            'MIN_TRACK_LENGTH': 16,
            'BEHAVIORS': ["stand", "lying_down", "foraging", "drinking_water", "rumination"]
        }

    def load_tracks(self, tracks_path: str) -> Dict[int, Dict[int, Tuple[float, float, float, float]]]:
        """
        Load MOT format tracks: frame,id,x,y,w,h,score,class,-1,-1
        Returns: {track_id: {frame_idx: (x,y,w,h)}}
        """
        tracks = defaultdict(dict)

        if not Path(tracks_path).exists():
            logger.error(f"Tracks file not found: {tracks_path}")
            return tracks

        logger.info(f"Loading tracks from {tracks_path}")

        with open(tracks_path, 'r') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    parts = line.split(',')
                    if len(parts) < 6:
                        logger.warning(f"Invalid line {line_idx}: {line}")
                        continue

                    frame_idx = int(parts[0])
                    track_id = int(parts[1])
                    x, y, w, h = map(float, parts[2:6])

                    tracks[track_id][frame_idx] = (x, y, w, h)

                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing line {line_idx}: {line} - {e}")
                    continue

        logger.info(f"Loaded {len(tracks)} tracks")
        return tracks

    def crop_and_resize(self, frame: np.ndarray, bbox: Tuple[float, float, float, float],
                       frame_width: int, frame_height: int) -> np.ndarray:
        """
        Crop region around bbox with padding and resize to OUT_SIZE x OUT_SIZE.

        Args:
            frame: Video frame (H, W, C)
            bbox: (x, y, w, h) bounding box
            frame_width, frame_height: Original frame dimensions

        Returns:
            Cropped and resized frame (OUT_SIZE, OUT_SIZE, C)
        """
        x, y, w, h = bbox

        # Add padding
        cx, cy = x + w/2, y + h/2
        w_pad, h_pad = w * (1 + self.PAD), h * (1 + self.PAD)

        # Calculate crop coordinates
        x1 = int(max(0, cx - w_pad/2))
        y1 = int(max(0, cy - h_pad/2))
        x2 = int(min(frame_width, cx + w_pad/2))
        y2 = int(min(frame_height, cy + h_pad/2))

        # Crop region
        roi = frame[y1:y2, x1:x2]

        # Handle edge cases
        if roi.size == 0:
            logger.warning(f"Empty ROI for bbox {bbox}")
            return np.zeros((self.OUT_SIZE, self.OUT_SIZE, 3), dtype=np.uint8)

        # Resize to target size
        resized = cv2.resize(roi, (self.OUT_SIZE, self.OUT_SIZE),
                           interpolation=cv2.INTER_AREA)

        return resized

    def extract_clips_from_track(self, video_path: str, track_id: int,
                                frames_dict: Dict[int, Tuple[float, float, float, float]]) -> List[Dict]:
        """
        Extract temporal clips from a single track.

        Args:
            video_path: Path to video file
            track_id: ID of the track
            frames_dict: {frame_idx: (x,y,w,h)} for this track

        Returns:
            List of clip metadata dictionaries
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"Processing track {track_id}: {len(frames_dict)} detections")

        # Sort frame indices
        frame_indices = sorted(frames_dict.keys())

        if len(frame_indices) < self.MIN_TRACK_LENGTH:
            logger.info(f"Track {track_id} too short ({len(frame_indices)} < {self.MIN_TRACK_LENGTH})")
            cap.release()
            return []

        clips_metadata = []

        # Sliding window over frames
        for i in range(0, len(frame_indices) - (self.T - 1) * self.STRIDE):
            # Get sequence indices
            seq_indices = [frame_indices[i + k * self.STRIDE] for k in range(self.T)]

            # Check temporal continuity
            if seq_indices[-1] - seq_indices[0] > self.STRIDE * (self.T - 1) + self.MAX_GAP:
                continue

            # Extract crops for this sequence
            crops = []
            valid_sequence = True

            for frame_idx in seq_indices:
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)  # OpenCV is 0-indexed
                ret, frame = cap.read()

                if not ret:
                    logger.warning(f"Cannot read frame {frame_idx}")
                    valid_sequence = False
                    break

                # Get bbox for this frame
                bbox = frames_dict[frame_idx]

                # Crop and resize
                crop = self.crop_and_resize(frame, bbox, width, height)

                # Convert BGR to RGB
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crops.append(crop_rgb)

            if valid_sequence and len(crops) == self.T:
                # Stack crops into (T, H, W, C) array
                clip_array = np.stack(crops, axis=0)

                # Create output directory for this track
                track_dir = self.clips_dir / f"id_{track_id:04d}"
                track_dir.mkdir(parents=True, exist_ok=True)

                # Save clip as .npy file
                clip_filename = f"{seq_indices[0]:06d}.npy"
                clip_path = track_dir / clip_filename
                np.save(clip_path, clip_array)

                # Create metadata
                clip_meta = {
                    "clip_path": str(clip_path),
                    "track_id": track_id,
                    "start_frame": seq_indices[0],
                    "end_frame": seq_indices[-1],
                    "frame_indices": seq_indices,
                    "video_path": video_path,
                    "fps": fps,
                    "timestamp_start": seq_indices[0] / fps,
                    "timestamp_end": seq_indices[-1] / fps,
                    "shape": clip_array.shape
                }

                clips_metadata.append(clip_meta)

        cap.release()
        logger.info(f"Extracted {len(clips_metadata)} clips from track {track_id}")
        return clips_metadata

    def process_video(self, video_path: str, tracks_path: str) -> List[Dict]:
        """
        Process a video and its corresponding tracks file.

        Args:
            video_path: Path to video file
            tracks_path: Path to MOT tracks file

        Returns:
            List of all clip metadata
        """
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Using tracks: {tracks_path}")

        # Load tracks
        tracks = self.load_tracks(tracks_path)

        if not tracks:
            logger.error("No tracks loaded, aborting")
            return []

        # Process each track
        all_clips_metadata = []

        for track_id, frames_dict in tracks.items():
            clip_metadata = self.extract_clips_from_track(video_path, track_id, frames_dict)
            all_clips_metadata.extend(clip_metadata)

        logger.info(f"Total clips extracted: {len(all_clips_metadata)}")
        return all_clips_metadata

    def save_metadata(self, clips_metadata: List[Dict], output_path: str = "workdir/clips_metadata.jsonl"):
        """Save clips metadata to JSONL file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for clip_meta in clips_metadata:
                f.write(json.dumps(clip_meta) + '\n')

        logger.info(f"Saved metadata for {len(clips_metadata)} clips to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract temporal clips from tracking data")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--tracks", required=True, help="Path to MOT tracks file")
    parser.add_argument("--config", default="configs/clip_maker.yaml",
                       help="Path to config file")
    parser.add_argument("--output", default="workdir/clips_metadata.jsonl",
                       help="Output metadata file")

    args = parser.parse_args()

    # Initialize clip maker
    clip_maker = ClipMaker(args.config)

    # Process video
    clips_metadata = clip_maker.process_video(args.video, args.tracks)

    # Save metadata
    if clips_metadata:
        clip_maker.save_metadata(clips_metadata, args.output)
        logger.info("Clip extraction completed successfully!")
    else:
        logger.error("No clips extracted!")


if __name__ == "__main__":
    main()