---
title: YOLOv8 Object Detection
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
pinned: true
license: mit
---

# 🎯 YOLOv8 Object Detection

Real-time CNN-based object detection using YOLOv8 with an interactive Gradio interface.

## Features
- Upload any image for instant object detection
- Adjustable confidence & IoU thresholds
- Class filtering (80 COCO classes)
- Bounding boxes with confidence scores
- Detailed detection table & JSON output

## Tech Stack
- **Model**: YOLOv8s (Ultralytics)
- **UI**: Gradio Blocks
- **Backbone**: CSPDarknet53 (CNN)
- **Dataset**: COCO (80 classes)
