# Cow Behavior Analysis with Computer Vision

**Internal Research Team Presentation | 10 Minutes**

---

## 1. Problem Statement & Approach (2 mins)

### The Challenge
- Manual cow behavior monitoring is labor-intensive and doesn't scale
- Need: Automated system for continuous behavior analysis from video footage

### Our Solution: Two-Stage Computer Vision Pipeline
```
Raw Video → YOLO Detection → ViT Classification → Behavior Analysis
```

### Dataset: CBVD-5
- **25,324** annotated bounding boxes across **537** video sequences
- **5 behavior classes**: Stand (33%), Rumination (24%), Foraging (23%), Lying down (18%), Drinking water (3%)

**FIGURE 1**: Show architecture diagram (Detection → Classification pipeline)

---

## 2. Key Methodological Innovation: Video-Based Splitting (1 min)

### The Problem with Random Splits
- Consecutive video frames are highly correlated
- Random splitting → temporal data leakage → inflated metrics

### Our Approach
- **Video-level splitting**: 375/107/55 videos (70/20/10% split)
- Extract video ID from filenames (e.g., `618_00002.jpg` → video `618`)
- Ensures no frames from same video appear in train/val/test

**Impact**: Realistic performance estimates for deployment

---

## 3. Model Performance (2 mins)

### Stage 1: YOLO Detection (YOLO11 nano)
- **Precision**: 86.4% | **Recall**: 84.8% | **mAP@50**: 90.6% | **mAP@50-95**: 50.1%
- Single-class cow detection optimized for speed/accuracy balance
- 640×640 resolution, 30 epochs training

### Stage 2: Vision Transformer Classification
- **Test Accuracy**: 92.42%
- **Weighted F1-Score**: 92.40%
- Fine-tuned from ImageNet-21k (86M parameters)

### Per-Class Performance
| Behavior | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| **Foraging** | 0.94 | **0.97** | 0.95 | 856 |
| **Stand** | **0.96** | 0.95 | 0.95 | 1,241 |
| Drinking water | 0.92 | 0.94 | 0.93 | 112 |
| Rumination | 0.88 | 0.90 | 0.89 | 912 |
| Lying down | 0.89 | 0.85 | 0.87 | 678 |

**FIGURE 2**: Confusion matrix showing minimal cross-class errors
**FIGURE 3**: Sample predictions with confidence scores

---

## 4. BotSort Multi-Object Tracking & Behavior Timelines (3 mins)

### Enhanced Pipeline: Individual Cow Tracking
- **BotSort integration**: Persistent ID tracking across video frames
- **Behavior timeline analysis**: Track time spent in each behavior per cow
- **Color-coded visualization**: Each cow gets unique color for easy tracking

### System Output Example
```
📊 Video Analysis Summary:
  Processed: 50 frames (2.0s @ 25 fps)
  Total detections: 187
  Unique cows tracked: 4

🐄 Individual Behavior Timelines:

  Track #1 (tracked for 1.8s):
    • foraging:     1.2s (67%)
    • stand:        0.6s (33%)

  Track #2 (tracked for 2.0s):
    • rumination:   1.5s (75%)
    • stand:        0.5s (25%)

  Track #3 (tracked for 1.5s):
    • lying down:   1.5s (100%)

  Track #4 (tracked for 1.2s):
    • stand:        0.9s (75%)
    • foraging:     0.3s (25%)
```

**FIGURE 4**: Show annotated video with BotSort tracking (color-coded IDs + behavior labels)

### Impact & Applications
- **Individual monitoring**: Track specific cows over time (not just herd-level)
- **Behavior patterns**: Quantify time budgets per animal
- **Health insights**: Detect deviations from normal behavior patterns
- **Scalability**: Foundation for long-term individual cow monitoring

---

## 5. Future Directions (1 min)

### Short-term
- **Unified evaluation framework**: Consolidate metrics from YOLO, ViT, and BotSort into single dashboard
  - End-to-end pipeline metrics (detection + classification + tracking)
  - Standardized benchmarking across model versions
  - Automated reporting and visualization
- **Temporal modeling**: Use LSTM/Transformer to leverage sequence information
- **Real-time optimization**: Model quantization and edge deployment

### Long-term
- **Methane production estimation**: Correlate rumination patterns with methane output
  - Improve rumination detection accuracy (currently 88% precision, 90% recall)
  - Model rumination duration and intensity
  - Research CV-based proxies for enteric methane emissions
- **Individual cow re-identification** across multiple videos
- **Health anomaly detection**: Flag unusual behavior patterns
- **Multi-farm deployment**: Federated learning while preserving privacy

---

## 6. Q&A (1 min)

---

# Recommended Figures to Collect

## Essential (Must Have)
1. **Architecture diagram** - Two-stage pipeline flow
2. **Confusion matrix** - From `05_vit_behavior_classifier.ipynb` cell 10
3. **BotSort demo video** - One of your `output_videos/tracked_*.mp4` files
4. **YOLO training curves** - `runs/detect/train5/results.png`

## Highly Recommended
5. **Class distribution bar chart** - Shows dataset imbalance
6. **Sample predictions grid** - From `05_vit_behavior_classifier.ipynb` (square 3x3 version)
7. **YOLO detection examples** - `runs/detect/train5/val_batch0_pred.jpg`
8. **Behavior timeline visualization** - Screenshot of terminal output showing individual cow timing data

## Nice to Have
9. **Detection confidence distribution** - Histogram of YOLO confidence scores
10. **Precision-Recall curves** - From `runs/detect/train5/BoxPR_curve.png`
11. **Training loss curves** - ViT training progression
12. **Temporal behavior transitions** - Heatmap showing behavior change patterns
