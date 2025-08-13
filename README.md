# Cow Behavior Analysis with YOLO + Vision Transformer

A complete machine learning pipeline for automated cow behavior classification using computer vision. This project combines YOLO object detection with Vision Transformer (ViT) classification to analyze cow behaviors in video footage.

## 📋 Project Overview

This repository implements an end-to-end system for:
- **Cow Detection**: Using YOLOv8 to detect and localize cows in video frames
- **Behavior Classification**: Using fine-tuned Vision Transformer to classify 5 cow behaviors
- **Pipeline Integration**: Complete workflow from raw video to annotated behavior analysis

### Key Results
- **Detection**: YOLOv8 nano model trained on 25K+ cow bounding boxes
- **Classification**: 92.6% accuracy on 5-class behavior classification
- **Pipeline**: Real-time video processing with frame-by-frame analysis

## 🗂️ Repository Structure

```
cow-sam/
├── 01_bbox_crops.ipynb           # Step 1: Extract crops from VIA annotations
├── 02_yolo_oneclass_from_via.ipynb  # Step 2: Train YOLO cow detector
├── 05_vit_behavior_classifier.ipynb # Step 3: Train ViT behavior classifier
├── 06_cow_detection_and_behavior_pipeline.ipynb # Step 4: End-to-end pipeline
├── README.md                     # This file
├── data/
│   ├── CBVD-5.csv               # VIA annotation file (25K+ annotations)
│   ├── labelframes/
│   │   └── labelframes/         # Video frame images (download required)
│   └── videos/
│       └── videos/              # Raw video files (download required)
├── models/
│   └── cow-behavior-vit/        # Trained ViT classifier (included)
├── workdir/
│   ├── crops_raw/               # Extracted behavior crops by class (generated)
│   └── yolo_cow_oneclass/       # YOLO training dataset (generated)
├── runs/                        # Training outputs and model weights (generated)
├── yolo11n.pt                   # Pre-trained YOLO weights (download required)
└── yolov8n.pt                   # Pre-trained YOLO weights (download required)
```

## 📊 Dataset Information

**CBVD-5 Dataset** (from [Kaggle](https://www.kaggle.com/datasets/fandaoerji/cbvd-5cow-behavior-video-dataset/data)):
- **Total Annotations**: 25,324 bounding box annotations
- **Video Sequences**: 537 unique video IDs
- **Behaviors**: 5 classes with the following distribution:
  - Stand: 8,272 (32.7%)
  - Rumination: 6,079 (24.0%)
  - Foraging: 5,711 (22.6%)
  - Lying down: 4,518 (17.8%)
  - Drinking water: 744 (2.9%)

**Annotation Format**: VIA (VGG Image Annotator) CSV format with spatial coordinates and behavior metadata.

### 🚨 Dataset Setup Required

**Important**: The large dataset files (~6GB) are excluded from this repository via `.gitignore`. 

#### 📋 **Manual Setup Required**

1. **Download the CBVD-5 dataset** from [Kaggle](https://www.kaggle.com/datasets/fandaoerji/cbvd-5cow-behavior-video-dataset/data)
2. **Extract the directories** from the downloaded zip file and place them in your `data/` folder:
   - Extract the entire `videos/` directory → place in `data/` (preserving nested structure)
   - Extract the entire `labelframes/` directory → place in `data/` (preserving nested structure)
   
   **Correct structure after extraction:**
   ```
   cow-sam/
   ├── data/
   │   ├── CBVD-5.csv          # ✅ Included (small metadata file)
   │   ├── videos/
   │   │   └── videos/         # ✅ Nested structure from dataset
   │   │       ├── video1.mp4
   │   │       ├── video2.mp4
   │   │       └── ...         # (~3.3GB, 687 total videos)
   │   └── labelframes/
   │       └── labelframes/    # ✅ Nested structure from dataset
   │           ├── image1.jpg
   │           ├── subfolder/
   │           └── ...         # (~2.7GB, 4,122 total images)
   └── yolo*.pt                # ❌ Download YOLO weights separately
   ```

3. **YOLO pre-trained weights** will be downloaded automatically when running the training notebooks.

The trained models in `models/` directory are included as they're much smaller and represent the key research outputs.

## 🚀 Notebook Execution Order

### Prerequisites
```bash
pip install ultralytics opencv-python numpy matplotlib scikit-learn pandas tqdm
pip install torch transformers datasets evaluate
```

### Step-by-Step Execution

#### 1. `01_bbox_crops.ipynb` - Extract Behavior Crops (START HERE!)
**Purpose**: Process VIA annotations to create padded bounding box crops organized by behavior class.

**Key Features**:
- Parses VIA CSV format annotations
- Applies behavior priority mapping (drinking > foraging > rumination > lying > standing)
- Extracts padded crops (8% padding) for better context
- Organizes crops into class-specific directories

**Output**: `workdir/crops_raw/` with 25K+ behavior-labeled image crops

**Runtime**: ~40 seconds for full dataset

---

#### 2. `02_yolo_oneclass_from_via.ipynb` - Train YOLO Detector
**Purpose**: Train YOLOv8 nano model for single-class cow detection using video-based data splitting.

**Key Design Choices**:
- **Video-based splitting** (70/20/10 train/val/test) to prevent data leakage
- **YOLOv8 nano** for speed/accuracy balance
- **Single class**: All cows treated as one class for detection
- **Data augmentation**: Built into YOLO training pipeline

**Technical Details**:
- 30 epochs training with early stopping
- 640x640 input resolution
- Mixed precision training (bf16/fp16)
- Video ID extraction from filenames for proper splitting

**Output**: Trained YOLO model at `runs/detect/train*/weights/best.pt`

**Performance**: Successfully detects cows across validation set

---

#### 3. `05_vit_behavior_classifier.ipynb` - Train Behavior Classifier
**Purpose**: Fine-tune Vision Transformer for 5-class cow behavior classification.

**Model Architecture**:
- **Base Model**: `google/vit-base-patch16-224-in21k`
- **Transfer Learning**: Pre-trained on ImageNet-21k, fine-tuned on cow behaviors
- **Input Size**: 224x224 RGB images
- **Classes**: 5 behaviors with custom label mapping

**Training Strategy**:
- **Stratified splitting**: Maintains class distribution across train/val/test
- **Mixed precision**: bf16 on supported hardware, fp16 fallback
- **Early stopping**: Patience=2 epochs based on weighted F1-score
- **Optimization**: AdamW with warmup and weight decay

**Key Results**:
- **Test Accuracy**: 92.6%
- **Weighted F1-Score**: 92.57%
- **Training Time**: ~30 minutes on RTX 4080

**Output**: Production-ready model saved to `models/cow-behavior-vit/`

---

#### 4. `06_cow_detection_and_behavior_pipeline.ipynb` - End-to-End Pipeline
**Purpose**: Integrate YOLO detection with ViT classification for complete video analysis.

**Pipeline Components**:
1. **Detection**: YOLO identifies cow bounding boxes
2. **Crop Extraction**: Extract regions of interest
3. **Classification**: ViT predicts behavior for each crop
4. **Visualization**: Annotated frames with behavior labels and confidence

**Features**:
- Real-time video processing
- Configurable confidence thresholds
- Frame-by-frame analysis with ffmpeg integration
- Visual output with bounding boxes and behavior labels

**Demo Capabilities**:
- Single image analysis
- Video processing with annotated output
- Sample validation on test images

## 🎯 Design Choices & Rationale

### 1. Video-Based Data Splitting
**Choice**: Split data by video ID rather than randomly
**Rationale**: Prevents data leakage since consecutive frames are highly correlated
**Implementation**: Extract video ID from filename pattern (e.g., `618_00002.jpg` → video `618`)

### 2. Behavior Priority Mapping
**Choice**: Hierarchical behavior assignment when multiple behaviors are present
**Priority Order**: drinking water > foraging > rumination > lying down > stand
**Rationale**: More specific/rare behaviors take precedence over common ones

### 3. Model Selection
**YOLO Choice**: YOLOv8 nano for detection
- **Pros**: Fast inference, good accuracy, single-shot detection
- **Trade-off**: Nano model for speed vs. accuracy balance

**ViT Choice**: `vit-base-patch16-224-in21k` for classification
- **Pros**: State-of-art vision model, excellent transfer learning
- **Trade-off**: Larger model size vs. superior accuracy

### 4. Data Augmentation Strategy
**Detection**: Relies on YOLO's built-in augmentation (rotation, scaling, color jittering)
**Classification**: Uses ViT's standard preprocessing (resize, normalize) without additional augmentation
**Rationale**: Large dataset size (25K+ samples) reduces need for aggressive augmentation

## 🔧 Future Improvements

### Short-term Enhancements
1. **Temporal Modeling**: Incorporate sequence information for behavior classification
2. **Multi-scale Detection**: Use multiple YOLO model sizes for accuracy/speed trade-offs
3. **Segmentation Integration**: Integrate SAM or similar segmentation model after detection to refine cow boundaries before classification
4. **Active Learning**: Implement uncertainty-based sampling for additional annotations
5. **Model Optimization**: Quantization and pruning for deployment efficiency

### Medium-term Developments
1. **Real-time Processing**: Optimize pipeline for live video streams
2. **Behavior Transition Analysis**: Track behavior changes over time
3. **Multi-animal Tracking**: Extend to track individual cow identities
4. **Environmental Context**: Incorporate location, time, and weather data

### Advanced Features
1. **3D Pose Estimation**: Add skeletal tracking for detailed behavior analysis
2. **Anomaly Detection**: Identify unusual behaviors or health issues
3. **Federated Learning**: Train across multiple farms while preserving privacy
4. **Mobile Deployment**: Develop smartphone/edge device applications

## 📈 Technical Performance

### YOLO Detection Model
- **Architecture**: YOLOv8 nano
- **Training**: 30 epochs with early stopping
- **Dataset**: 3,199 annotated images (video-based split)
- **Performance**: Reliable cow detection across diverse conditions

### ViT Classification Model
- **Architecture**: ViT-base-patch16-224 (86M parameters)
- **Training**: 10 epochs with early stopping
- **Dataset**: 25,324 behavior crops (stratified split)
- **Results**:
  ```
  Test Accuracy: 92.6%
  Weighted F1-Score: 92.57%
  
  Per-class Performance:
  - drinking water: 95% precision, 89% recall
  - foraging: 91% precision, 94% recall  
  - lying down: 94% precision, 91% recall
  - rumination: 93% precision, 92% recall
  - stand: 92% precision, 95% recall
  ```