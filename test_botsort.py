#!/usr/bin/env python3
"""
BotSort Tracking Test Script
Demonstrates BotSort tracker functionality with YOLO model
"""

from ultralytics import YOLO
from pathlib import Path

def test_botsort_availability():
    """Test if BotSort is available in ultralytics"""
    print("=" * 60)
    print("BotSort Availability Test")
    print("=" * 60)

    try:
        from ultralytics.trackers.bot_sort import BOTSORT
        print("✓ BOTSORT class imported successfully")

        # Check config file
        import ultralytics
        config_path = Path(ultralytics.__file__).parent / "cfg/trackers/botsort.yaml"
        if config_path.exists():
            print(f"✓ BotSort config found: {config_path}")
        else:
            print(f"✗ BotSort config not found")
            return False

        return True
    except ImportError as e:
        print(f"✗ Failed to import BOTSORT: {e}")
        return False

def test_tracking_with_image():
    """Test tracking functionality with a sample image"""
    print("\n" + "=" * 60)
    print("BotSort Tracking Test")
    print("=" * 60)

    # Check if trained model exists
    model_path = Path("runs/detect/train3/weights/best.pt")
    if not model_path.exists():
        print(f"⚠ Model not found at {model_path}")
        print("Using YOLOv8n pretrained model instead for demo")
        model_path = "yolo11n.pt"

    # Initialize model
    print(f"\nLoading model: {model_path}")
    model = YOLO(str(model_path))

    # Find a test image
    test_image = None
    image_dirs = [
        Path("workdir/yolo_cow_oneclass/images/val"),
        Path("data/labelframes/labelframes"),
    ]

    for img_dir in image_dirs:
        if img_dir.exists():
            images = list(img_dir.glob("*.jpg"))[:1]
            if images:
                test_image = images[0]
                break

    if test_image is None:
        print("⚠ No test images found. Skipping tracking demo.")
        print("To test tracking, add images to one of these directories:")
        for d in image_dirs:
            print(f"  - {d}")
        return

    print(f"\nTest image: {test_image.name}")

    # Test 1: Regular detection (no tracking)
    print("\n--- Test 1: Regular Detection (no tracking) ---")
    results = model.predict(source=str(test_image), conf=0.25, verbose=False)
    print(f"Detected {len(results[0].boxes)} objects")

    # Test 2: Detection with BotSort tracking
    print("\n--- Test 2: BotSort Tracking ---")
    try:
        results = model.track(
            source=str(test_image),
            tracker='botsort.yaml',
            conf=0.25,
            persist=True,
            verbose=False
        )
        print(f"✓ BotSort tracking successful!")
        print(f"  Tracked {len(results[0].boxes)} objects")

        # Show track IDs if available
        if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.cpu().numpy()
            print(f"  Track IDs: {track_ids}")

    except Exception as e:
        print(f"✗ Tracking failed: {e}")
        return False

    print("\n✓ All tracking tests passed!")
    return True

def print_botsort_config():
    """Display BotSort configuration"""
    print("\n" + "=" * 60)
    print("BotSort Configuration")
    print("=" * 60)

    import ultralytics
    config_path = Path(ultralytics.__file__).parent / "cfg/trackers/botsort.yaml"

    if config_path.exists():
        with open(config_path) as f:
            print(f.read())
    else:
        print("Config file not found")

if __name__ == "__main__":
    print("\nBotSort Integration Test for Cow Behavior Analysis")
    print("Ultralytics YOLO + BotSort Tracker\n")

    # Test 1: Check availability
    if not test_botsort_availability():
        print("\n✗ BotSort not available. Exiting.")
        exit(1)

    # Test 2: Show configuration
    print_botsort_config()

    # Test 3: Test tracking
    test_tracking_with_image()

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    print("\nTo use BotSort in your notebooks:")
    print("  model = YOLO('path/to/model.pt')")
    print("  results = model.track(source='video.mp4', tracker='botsort.yaml')")
    print("\nFor ReID (re-identification), edit botsort.yaml:")
    print("  with_reid: True")
    print("=" * 60)
