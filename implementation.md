# Implementation Guide

## SawitVision: Palm Tree Detection and Counting from Aerial Imagery

Technical blueprint and architecture guide for the palm tree detection, counting, and geospatial analysis pipeline. Accompanies [PRD.md](PRD.md).

> **Scope Note:** SawitVision is built using open-source frameworks, open academic datasets, and custom algorithms. All parameters and hyperparameter configurations serve as modular defaults designed to be tuned across specific flight profiles.

---

## 1. System Architecture

```text
 Aerial Imagery (GeoTIFF / JPG / PNG)
                 │
                 ▼
     ┌────────────────────────┐
     │  Input Preprocessing   │  Normalization, CRS extraction, resolution check
     └───────────┬────────────┘
                 ▼
     ┌────────────────────────┐
     │ Sliced Window Tiling   │  Configurable stride & spatial overlap
     └───────────┬────────────┘
                 ▼
     ┌────────────────────────┐
     │ PyTorch Object Detector│  Deep learning crown detection per window
     └───────────┬────────────┘
                 ▼
     ┌────────────────────────┐
     │ Deduplication & Merge  │  Cross-window NMS / GreedNMM merge
     └───────────┬────────────┘
                 ▼
     ┌────────────────────────┐
     │ Post-Processing Engine │  Deterministic numbering (1..N), pixel ↔ map coordinates,
     └───────────┬────────────┘  per-block containment, uncertainty crop extraction
                 │
  ┌──────────────┴───────────────────────────────────┐
  │                                                  │
  ▼                                                  ▼
Structured Artifacts:                              Service Layer:
- Numbered High-Res Preview                        - Asynchronous REST API (FastAPI)
- CSV Stand Roster                                 - Interactive Web Viewer
- GeoJSON Vector Centroids                         - Downstream Integration Endpoints
- Density Grids & Uncertainty Crops
```

---

## 2. Technology Stack

| Layer | Component | Technical Role |
|---|---|---|
| Deep Learning Engine | PyTorch & Modern YOLO Architecture | High-speed, anchor-free single-class (`oil_palm`) object detection |
| Sliced Inference | Sliced Windowing Engine (SAHI-compatible) | High-resolution inference without loss of fine crown detail |
| Geospatial Processing | Shapely, Rasterio, GeoPandas, PyProj | Coordinate reference transformation, polygon containment, vector exports |
| Computer Vision & Ops | OpenCV, NumPy, Pillow, Matplotlib | Image manipulation, HSV canopy analysis, visual rendering |
| Annotation & Curation | Open-source tools (CVAT, Label Studio) | YOLO-format bounding box annotation |
| Experiment Tracking | MLflow / Structured Run Logging | Hyperparameter tracking, validation mAP, and counting metrics |
| Service & API | FastAPI + Uvicorn | Asynchronous job submission, status polling, and artifact retrieval |
| Deployment & Edge | Docker, ONNX Runtime | Containerized microservice and CPU-accelerated inference |

---

## 3. Repository Structure

```text
sawit-vision/
├── README.md
├── PRD.md
├── ROADMAP.md
├── implementation.md
├── requirements.txt
├── configs/
│   ├── data.yaml              # Dataset paths and class mapping
│   └── train.yaml             # Training and augmentation hyperparameters
├── data/                      # Excluded from git (download scripts only)
│   ├── raw/                   # Raw drone and aerial imagery
│   ├── samples/               # Representative demo images
│   ├── tiles/                 # Sliced window tiles for training
│   └── splits/                # Location-separated train/val/test splits
├── scripts/
│   ├── detect_and_number.py   # Baseline detection, numbering (1..N), and visualization
│   ├── crop_and_classify.py   # Crop extraction and canopy attribute analysis
│   ├── download_data.py       # Automated fetch for open academic benchmarks
│   ├── make_tiles.py          # Windowed slicing for large rasters
│   ├── train.py               # Model training script
│   └── evaluate.py            # Comprehensive detection and count metrics suite
├── src/sawitvision/
│   ├── __init__.py
│   ├── detector.py            # Model wrapper and inference interface
│   ├── infer.py               # Sliced sliding window execution and box merging
│   ├── geo.py                 # Pixel-to-geographic coordinate transformations & GeoJSON
│   ├── blocks.py              # Management plot polygon containment and trees/ha aggregation
│   ├── viz.py                 # Deterministic numbering renderer, heatmaps, and audit crops
│   ├── metrics.py             # MAE, MAPE, bias, R² count evaluation calculations
│   └── api/
│       ├── main.py            # FastAPI service application
│       ├── schemas.py         # Request and response Pydantic models
│       └── jobs.py            # Background job management
├── tests/
│   ├── test_metrics.py        # Count and detection metric validation
│   ├── test_geo.py            # Spatial projection tests
│   └── test_pipeline.py       # End-to-end integration tests
└── docs/
    ├── data_sources.md        # Comprehensive dataset provenance and licenses
    └── model_card.md          # Intended use, operating bounds, and failure modes
```

---

## 4. Data Engineering and Processing

### 4.1 Annotation Standard

Standardized YOLO bounding box format with normalized coordinates `[0, 1]`:

```text
<class_id> <x_center> <y_center> <width> <height>
```
Single target class: `0 = oil_palm`.

For datasets providing point-only labels (e.g. canopy centers), a fixed circular/square bounding radius calibrated to mean crown diameter (e.g. 5–8 meters ground distance) is generated programmatically.

### 4.2 Sliced Windowing for Training and Inference

High-resolution drone orthomosaics cannot be naively downscaled without losing distinguishing frond structures.
- Window sizes: 640×640 or 1024×1024 pixels.
- Overlap ratio: 20% horizontal and vertical overlap to guarantee boundary coverage.
- Edge filtering: Filter out boxes clipped below 40% visible area during tile generation to avoid training on truncated frond segments.

### 4.3 Preventing Data Leakage via Spatial Splits

Do not use uniform random tile splitting. Slices from the same flight pass share identical lighting and canopy structure.
- **Train Split (~70%):** Distinct geographic sectors.
- **Validation Split (~15%):** Separate flight blocks for parameter selection.
- **Test Split (~15%):** Held-out geographic zone with distinct topography and lighting conditions.

---

## 5. Evaluation Framework

Both detection accuracy and counting integrity must be tracked simultaneously:

### 5.1 Detection Metrics
- Precision, Recall, and F1 score at IoU = 0.5.
- Mean Average Precision: mAP@0.5 and mAP@0.5:0.95.

### 5.2 Count Metrics (Per Image and Per Management Plot)
- **Mean Absolute Error (MAE):** $\frac{1}{N}\sum |y_{pred} - y_{true}|$
- **Mean Absolute Percentage Error (MAPE):** $\frac{1}{N}\sum \frac{|y_{pred} - y_{true}|}{\max(y_{true}, 1)}$
- **Systematic Bias:** $\frac{1}{N}\sum (y_{pred} - y_{true})$ (detects persistent under/overcounting tendencies)
- **Coefficient of Determination ($R^2$):** Quantifies variance explained across varying plot densities.

---

## 6. Deterministic Crown Numbering and Review Workflows

1. **Deterministic Ordering (FR-4):**
   Crown detections are sorted deterministically from top-to-bottom across horizontal bands, then left-to-right. Each detection receives a permanent index $1 \dots N$.
2. **Visual Preview:**
   High-contrast bounding boxes and index tags are drawn onto the output preview with customizable background halos to ensure legibility across bright foliage and deep shadow.
3. **Audit Crop Extraction (FR-12 & FR-15):**
   Detections with borderline confidence scores ($0.25 \le conf < 0.50$) or anomalous aspect ratios are extracted as cropped image chips for agronomist verification.

---

## 7. Geospatial & Boundary Aggregation

- **Projection Pipeline:** Reads affine transformation matrices and coordinate reference systems (CRS) from GeoTIFF headers. Maps pixel centroids $(c_x, c_y)$ into real-world coordinates (e.g., EPSG:4326 or local UTM projections).
- **Block Aggregation:** Point-in-polygon spatial join computes exact tree counts, canopy density (trees/ha), and identifies stand vacancies per management block.

---

## 8. Deployment and API Design

- **API Architecture:** Asynchronous job pattern (`/v1/jobs`) powered by FastAPI. Allows processing high-resolution aerial files without HTTP timeouts.
- **Containerization:** Self-contained Docker environment with multi-stage builds and automated health checks.
- **Edge Deployment:** Export pipeline converting trained PyTorch weights to ONNX for CPU-only inference on field laptops.

---

*Author: Ryan Radityatama · [ryradit.my.id](https://ryradit.my.id)*
