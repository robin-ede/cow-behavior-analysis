## Dataset Notes (CBVD-5)

This document captures practical context for the CBVD-5 dataset used in this repository.
It is intended as a quick reference for reproducibility and paper writing.

## Origin and Scope

- Source dataset: CBVD-5 (cow behavior video dataset)
- Data collection context: Inner Mongolia Autonomous Region, China
- Goal: benchmark dataset for behavior recognition in dairy-barn conditions

## Capture Environment

- Setting: standardized indoor dairy barn
- Population: 107 dairy cows (about 20 months old)
- Observation duration: 96 hours continuous monitoring
- Camera placement covers feeding, watering, and resting areas
- Environment is indoor, so weather variation is limited within the scene

## Lighting and Video Collection

- Daytime and nighttime conditions are represented
- Fixed infrared cameras were used for 24/7 monitoring
- Reported camera/lens setup includes 2.8 mm and 3.6 mm lenses

## Behavior Labels

The five behavior categories used in this repo are:

1. stand
2. lying down
3. foraging
4. drinking water
5. rumination

## Annotation and Format

- Recorded from 7 fixed cameras
- Reported totals include:
  - 687 video segments
  - about 206,100 image samples
  - 4,122 manually labeled keyframes (VIA 3.0.11)
  - 27,501 valid labeled instances after review
- Labels are AVA-style action annotations with per-box behavior IDs

## Known Quality Constraints

- Class imbalance exists (especially drinking water)
- Behavior overlap is non-trivial:
  - lying down vs rumination may co-occur
  - head-down postures may confuse foraging vs drinking
- Reported error modes include missed detections, missed boxes, and false boxes

## Breed Note

The original description does not clearly specify breed in all summaries.
Treat breed as dairy-cattle context unless a paper citation explicitly confirms a breed label.

## Repo-Specific Usage Notes

- Large raw assets are intentionally not committed; place dataset files under `data/`
- Expected paths used by notebooks/scripts:
  - `data/CBVD-5.csv`
  - `data/labelframes/labelframes/`
  - `data/videos/videos/`
- Generated artifacts should go under `workdir/` and `artifacts/`

## Quick Summary Table

| Aspect | Details |
| --- | --- |
| Dataset | CBVD-5 |
| Region | Inner Mongolia, China |
| Cows | 107 dairy cows, approx. 20 months |
| Duration | 96 hours continuous |
| Videos | 687 segments |
| Images | approx. 206,100 samples |
| Labeled keyframes | 4,122 |
| Valid labeled instances | 27,501 |
| Classes | 5 behaviors |
| Annotation tool | VIA 3.0.11 |
