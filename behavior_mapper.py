#!/usr/bin/env python3
"""
Behavior Label Mapper for Cow Behavior Analysis

Maps temporal clips to behavior labels using VIA CSV annotations.
Handles temporal overlap and majority voting for clip labeling.

Usage:
    python behavior_mapper.py --clips workdir/clips_metadata.jsonl --via data/CBVD-5.csv --output workdir/clips_labeled.jsonl
"""

import argparse
import csv
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BehaviorMapper:
    def __init__(self):
        """Initialize BehaviorMapper."""
        # VIA behavior mapping (from SPEC.md)
        self.via_behaviors = {
            "0": "stand",
            "1": "lying_down",
            "2": "foraging",
            "3": "drinking_water",
            "4": "rumination"
        }

        # Priority order for conflicting labels (drinking > foraging > rumination > lying > stand)
        self.behavior_priority = {
            "drinking_water": 5,
            "foraging": 4,
            "rumination": 3,
            "lying_down": 2,
            "stand": 1
        }

    def parse_via_csv(self, via_path: str) -> List[Dict]:
        """
        Parse VIA CSV file to extract behavior annotations.

        Args:
            via_path: Path to VIA CSV file

        Returns:
            List of annotation dictionaries
        """
        annotations = []

        logger.info(f"Parsing VIA CSV: {via_path}")

        with open(via_path, 'r') as f:
            # Skip header comments
            lines = []
            for line in f:
                if not line.startswith('#'):
                    lines.append(line)

        # Parse CSV
        reader = csv.DictReader(lines)

        for row in reader:
            try:
                # Extract temporal coordinates
                temporal_coords = row['temporal_coordinates'].strip('[]')
                if ',' in temporal_coords:
                    # Temporal segment [start, end]
                    start_time, end_time = map(float, temporal_coords.split(','))
                else:
                    # Single frame timestamp
                    start_time = end_time = float(temporal_coords)

                # Extract spatial coordinates [shape_id, x, y, w, h]
                spatial_coords = row['spatial_coordinates'].strip('[]')
                shape_parts = list(map(float, spatial_coords.split(',')))

                if len(shape_parts) >= 5:
                    shape_id, x, y, w, h = shape_parts[:5]

                    # Extract behavior from metadata
                    metadata = row['metadata']
                    if metadata and '1' in metadata:
                        # Parse metadata like {"1":"3"}
                        metadata_dict = json.loads(metadata.replace("'", '"'))
                        behavior_id = metadata_dict.get("1", "0")
                        behavior = self.via_behaviors.get(behavior_id, "stand")
                    else:
                        behavior = "stand"  # default

                    # Get video filename
                    file_list = row['file_list']

                    annotation = {
                        'file': file_list,
                        'start_time': start_time,
                        'end_time': end_time,
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'behavior': behavior,
                        'shape_id': shape_id
                    }

                    annotations.append(annotation)

            except (ValueError, KeyError, json.JSONDecodeError) as e:
                logger.warning(f"Error parsing annotation row: {e}")
                continue

        logger.info(f"Parsed {len(annotations)} annotations")
        return annotations

    def get_overlapping_annotations(self, clip_meta: Dict, annotations: List[Dict],
                                  overlap_threshold: float = 0.5) -> List[Dict]:
        """
        Find annotations that overlap with a clip temporally and spatially.

        Args:
            clip_meta: Clip metadata with timestamp and track info
            annotations: List of VIA annotations
            overlap_threshold: Minimum overlap ratio for spatial matching

        Returns:
            List of overlapping annotations
        """
        clip_start = clip_meta['timestamp_start']
        clip_end = clip_meta['timestamp_end']
        video_file = Path(clip_meta['video_path']).name

        overlapping = []

        for ann in annotations:
            # Check if same video file
            if ann['file'] != video_file:
                continue

            # Check temporal overlap
            ann_start, ann_end = ann['start_time'], ann['end_time']

            # Calculate temporal overlap
            overlap_start = max(clip_start, ann_start)
            overlap_end = min(clip_end, ann_end)

            if overlap_start < overlap_end:
                # Temporal overlap exists
                temporal_overlap = (overlap_end - overlap_start) / (clip_end - clip_start)

                if temporal_overlap >= 0.1:  # At least 10% temporal overlap
                    overlapping.append({
                        **ann,
                        'temporal_overlap': temporal_overlap
                    })

        return overlapping

    def assign_clip_behavior(self, overlapping_annotations: List[Dict]) -> str:
        """
        Assign behavior to clip based on overlapping annotations.
        Uses majority voting with priority weighting.

        Args:
            overlapping_annotations: List of annotations overlapping with clip

        Returns:
            Assigned behavior label
        """
        if not overlapping_annotations:
            return "stand"  # default

        # Weight by temporal overlap and behavior priority
        behavior_scores = defaultdict(float)

        for ann in overlapping_annotations:
            behavior = ann['behavior']
            temporal_weight = ann['temporal_overlap']
            priority_weight = self.behavior_priority.get(behavior, 1)

            score = temporal_weight * priority_weight
            behavior_scores[behavior] += score

        # Return behavior with highest score
        best_behavior = max(behavior_scores.items(), key=lambda x: x[1])[0]
        return best_behavior

    def map_clips_to_behaviors(self, clips_metadata: List[Dict],
                              annotations: List[Dict]) -> List[Dict]:
        """
        Map all clips to behavior labels.

        Args:
            clips_metadata: List of clip metadata dictionaries
            annotations: List of VIA annotations

        Returns:
            List of clips with assigned behavior labels
        """
        labeled_clips = []

        logger.info(f"Mapping {len(clips_metadata)} clips to behaviors")

        behavior_counts = Counter()

        for clip_meta in clips_metadata:
            # Find overlapping annotations
            overlapping = self.get_overlapping_annotations(clip_meta, annotations)

            # Assign behavior
            behavior = self.assign_clip_behavior(overlapping)
            behavior_counts[behavior] += 1

            # Add behavior to clip metadata
            labeled_clip = {
                **clip_meta,
                'behavior': behavior,
                'num_annotations': len(overlapping),
                'annotation_confidence': len(overlapping) / max(1, len(overlapping))
            }

            labeled_clips.append(labeled_clip)

        # Log behavior distribution
        logger.info("Behavior distribution:")
        for behavior, count in behavior_counts.most_common():
            logger.info(f"  {behavior}: {count} clips")

        return labeled_clips

    def add_train_test_splits(self, labeled_clips: List[Dict],
                            split_ratios: Dict[str, float] = None) -> List[Dict]:
        """
        Add train/val/test splits based on track IDs (identity-safe).

        Args:
            labeled_clips: List of labeled clips
            split_ratios: Dict with train/val/test ratios

        Returns:
            List of clips with split assignments
        """
        if split_ratios is None:
            split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}

        # Group by track ID
        track_clips = defaultdict(list)
        for clip in labeled_clips:
            track_clips[clip['track_id']].append(clip)

        # Assign tracks to splits
        track_ids = list(track_clips.keys())
        track_ids.sort()  # For reproducibility

        n_tracks = len(track_ids)
        n_train = int(n_tracks * split_ratios['train'])
        n_val = int(n_tracks * split_ratios['val'])

        train_tracks = track_ids[:n_train]
        val_tracks = track_ids[n_train:n_train + n_val]
        test_tracks = track_ids[n_train + n_val:]

        # Assign splits to clips
        for clip in labeled_clips:
            track_id = clip['track_id']
            if track_id in train_tracks:
                clip['split'] = 'train'
            elif track_id in val_tracks:
                clip['split'] = 'val'
            else:
                clip['split'] = 'test'

        # Log split statistics
        split_counts = Counter(clip['split'] for clip in labeled_clips)
        logger.info("Split distribution:")
        for split, count in split_counts.items():
            logger.info(f"  {split}: {count} clips")

        return labeled_clips

    def process(self, clips_metadata_path: str, via_csv_path: str,
               output_path: str) -> None:
        """
        Complete processing pipeline.

        Args:
            clips_metadata_path: Path to clips metadata JSONL
            via_csv_path: Path to VIA CSV annotations
            output_path: Output path for labeled clips
        """
        # Load clips metadata
        logger.info(f"Loading clips metadata from {clips_metadata_path}")
        clips_metadata = []
        with open(clips_metadata_path, 'r') as f:
            for line in f:
                clips_metadata.append(json.loads(line.strip()))

        # Parse VIA annotations
        annotations = self.parse_via_csv(via_csv_path)

        # Map clips to behaviors
        labeled_clips = self.map_clips_to_behaviors(clips_metadata, annotations)

        # Add train/test splits
        labeled_clips = self.add_train_test_splits(labeled_clips)

        # Save labeled clips
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for clip in labeled_clips:
                f.write(json.dumps(clip) + '\n')

        logger.info(f"Saved {len(labeled_clips)} labeled clips to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Map clips to behavior labels")
    parser.add_argument("--clips", required=True,
                       help="Path to clips metadata JSONL file")
    parser.add_argument("--via", required=True,
                       help="Path to VIA CSV annotations file")
    parser.add_argument("--output", default="workdir/clips_labeled.jsonl",
                       help="Output path for labeled clips")

    args = parser.parse_args()

    # Initialize mapper
    mapper = BehaviorMapper()

    # Process
    mapper.process(args.clips, args.via, args.output)
    logger.info("Behavior mapping completed successfully!")


if __name__ == "__main__":
    main()