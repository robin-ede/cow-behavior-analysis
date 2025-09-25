#!/usr/bin/env python3
"""
Test script for clip_maker.py

Creates sample MOT tracks and tests clip extraction functionality.
"""

import os
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_tracks():
    """Create a sample MOT tracks file for testing."""
    tracks_path = Path("workdir/mot_tracks/sample_video.txt")
    tracks_path.parent.mkdir(parents=True, exist_ok=True)

    # Create sample tracks: frame,id,x,y,w,h,score,class,-1,-1
    # Track 1: moves from left to right
    # Track 2: stays in center
    sample_tracks = []

    # Track 1: 50 frames, moving cow
    for frame in range(1, 51):
        x = 100 + frame * 2  # moving right
        y = 200
        w, h = 80, 120
        score = 0.9
        sample_tracks.append(f"{frame},1,{x},{y},{w},{h},{score},0,-1,-1")

    # Track 2: 30 frames, stationary cow (overlapping time)
    for frame in range(20, 50):
        x = 400
        y = 150
        w, h = 90, 110
        score = 0.85
        sample_tracks.append(f"{frame},2,{x},{y},{w},{h},{score},0,-1,-1")

    # Track 3: shorter track (should be filtered out)
    for frame in range(40, 45):
        x = 600
        y = 300
        w, h = 70, 100
        score = 0.8
        sample_tracks.append(f"{frame},3,{x},{y},{w},{h},{score},0,-1,-1")

    with open(tracks_path, 'w') as f:
        for track_line in sample_tracks:
            f.write(track_line + '\n')

    logger.info(f"Created sample tracks file: {tracks_path}")
    logger.info(f"Track 1: 50 frames, Track 2: 30 frames, Track 3: 5 frames")
    return str(tracks_path)

def find_sample_video():
    """Find an existing video file for testing."""
    video_extensions = ['.mp4', '.avi', '.mov']

    # Check output_videos directory
    output_dir = Path("output_videos")
    if output_dir.exists():
        for ext in video_extensions:
            videos = list(output_dir.glob(f"*{ext}"))
            if videos:
                return str(videos[0])

    # Check data directory (though unlikely to have videos there)
    data_dir = Path("data")
    if data_dir.exists():
        for ext in video_extensions:
            videos = list(data_dir.glob(f"*{ext}"))
            if videos:
                return str(videos[0])

    logger.warning("No video files found for testing")
    return None

def test_clip_maker():
    """Test the clip maker functionality."""
    logger.info("Testing ClipMaker...")

    # Import the clip maker
    try:
        sys.path.insert(0, '.')
        exec(open('04_clip_maker.py').read())
        from clip_maker import ClipMaker
    except:
        # Alternative method - import directly
        import importlib.util
        spec = importlib.util.spec_from_file_location("clip_maker", "04_clip_maker.py")
        clip_maker_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(clip_maker_module)
        ClipMaker = clip_maker_module.ClipMaker

    # Create sample tracks
    tracks_path = create_sample_tracks()

    # Find a sample video
    video_path = find_sample_video()
    if not video_path:
        logger.error("No video file found for testing. Please add a video file to output_videos/")
        return False

    logger.info(f"Using video: {video_path}")

    # Initialize clip maker
    clip_maker = ClipMaker()

    # Test loading tracks
    tracks = clip_maker.load_tracks(tracks_path)
    logger.info(f"Loaded {len(tracks)} tracks")

    for track_id, frames in tracks.items():
        logger.info(f"Track {track_id}: {len(frames)} frames")

    # Test clip extraction (dry run - just metadata)
    logger.info("Testing clip extraction...")

    try:
        clips_metadata = clip_maker.process_video(video_path, tracks_path)

        if clips_metadata:
            logger.info(f"Successfully extracted {len(clips_metadata)} clips")

            # Show some statistics
            track_counts = {}
            for clip in clips_metadata:
                track_id = clip['track_id']
                track_counts[track_id] = track_counts.get(track_id, 0) + 1

            logger.info("Clips per track:")
            for track_id, count in track_counts.items():
                logger.info(f"  Track {track_id}: {count} clips")

            # Save metadata
            clip_maker.save_metadata(clips_metadata, "workdir/test_clips_metadata.jsonl")

            return True
        else:
            logger.error("No clips extracted")
            return False

    except Exception as e:
        logger.error(f"Error during clip extraction: {e}")
        return False

def test_behavior_mapper():
    """Test the behavior mapper functionality."""
    logger.info("Testing BehaviorMapper...")

    # Check if we have the required files
    clips_metadata_path = "workdir/test_clips_metadata.jsonl"
    via_csv_path = "data/CBVD-5.csv"

    if not Path(clips_metadata_path).exists():
        logger.warning("No clips metadata found, skipping behavior mapping test")
        return False

    if not Path(via_csv_path).exists():
        logger.warning("No VIA CSV found, skipping behavior mapping test")
        return False

    try:
        from behavior_mapper import BehaviorMapper

        mapper = BehaviorMapper()

        # Test VIA parsing
        annotations = mapper.parse_via_csv(via_csv_path)
        logger.info(f"Parsed {len(annotations)} VIA annotations")

        # Test full processing
        mapper.process(clips_metadata_path, via_csv_path, "workdir/test_clips_labeled.jsonl")

        return True

    except Exception as e:
        logger.error(f"Error during behavior mapping: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting ClipMaker tests...")

    # Test clip maker
    clip_success = test_clip_maker()

    # Test behavior mapper if clip maker succeeded
    behavior_success = False
    if clip_success:
        behavior_success = test_behavior_mapper()

    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    logger.info(f"ClipMaker: {'PASS' if clip_success else 'FAIL'}")
    logger.info(f"BehaviorMapper: {'PASS' if behavior_success else 'SKIP/FAIL'}")

    if clip_success:
        logger.info("\nNext steps:")
        logger.info("1. Run tracking (03_tracking_ocsort_bytetrack.ipynb) to generate real MOT files")
        logger.info("2. Use 04_clip_maker.py to extract clips from real tracking data")
        logger.info("3. Use behavior_mapper.py to assign behavior labels")
        logger.info("4. Train X3D temporal classifier (05_x3d_train.py)")

if __name__ == "__main__":
    main()