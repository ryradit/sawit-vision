# SawitVision: Palm Tree Detection, Stand Inventory, and Geospatial Analytics

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![TorchScript](https://img.shields.io/badge/Deploy-TorchScript-orange.svg)](models/)

An automated computer vision and geospatial analytics system for precision palm tree detection, sequential crown numbering ($1 \dots N$), stand density estimation, and vacancy detection from high-resolution aerial and drone orthomosaics.

---

## Key Highlights

- **Automated Stand Counting & Numbering:** Numbers each detected palm crown sequentially from $1 \dots N$ across row bands for field auditability.
- **Sliced Sliding-Window Inference (SAHI Engine):** Ingests gigapixel orthomosaics with border deduplication via Greedy-NMM (Intersection-Over-Smaller).
- **Multi-Condition Benchmark Testing:** Tested across 8 benchmark scenarios covering dense interlocking canopies, severe shadow gradients, boundary transitions, chlorotic/stressed foliage, and altitude shifts.
- **Fine-Tuned Detection Weights:** Includes fine-tuned PyTorch detection checkpoints (`models/best.pt`) and zero-overhead edge deployment binaries (`models/best.torchscript`).
- **Agronomic Management Block Intelligence:** Computes observed vs. target stand density (trees/ha), planting grid regularity (Clark-Evans $R$), and localizes missing stand vacancies.
- **Automated Metrics Evaluation Dashboards:** Generates multi-panel analytical graph dashboards tracking confidence distribution, count sensitivity, inter-tree spacing, and spatial density.
- **Comprehensive Open Dataset Catalog:** Curated catalog of 23 open-access drone, aerial, and satellite benchmarks across Southeast Asia and global tropical agriculture.

---

## Architecture Overview

```text
Aerial Orthomosaic (GeoTIFF / JPG / PNG)
                   │
                   ▼
     [ Sliced Window Tiling ]  ──► (640x640 px, 20% overlap)
                   │
                   ▼
     [ PyTorch / YOLO Detector ] ──► (models/best.pt or Adaptive Baseline)
                   │
                   ▼
     [ Cross-Tile Deduplication ] ──► (Greedy-NMM IoS Merge)
                   │
                   ▼
     [ Deterministic Numbering ]  ──► (Sequential IDs 1..N)
                   │
   ┌───────────────┴────────────────────────────────────────┐
   ▼                                                        ▼
[ Spatial & Agronomic Intelligence ]         [ Structured Visual Artifacts ]
- Management Block Containment               - Numbered Audit Preview (.jpg)
- Stand Density (trees/ha)                   - Stand Roster (.csv)
- Missing Tree Vacancy Detection             - Spatial Point Layer (.geojson)
- Regularity Score (Clark-Evans R)           - Metrics Dashboard Graph (.png)
```

---

## Multi-Condition Generalization Benchmark Results

Tested across 8 distinct environmental and flight conditions using sliced sliding-window inference:

| Benchmark Condition | Input Resolution | Stand Count | Spacing ($\mu$) | Spacing CV | Regularity ($R$) | Tested Scenario |
|---|---|---|---|---|---|---|
| **Dense Overlapping Canopy** | $600 \times 500\text{ px}$ | **242 palms** | $2.54\text{ m}$ | $0.06$ | **$1.81$** | Interlocking fronds in interior commercial block |
| **Shadow & Low Sun Angle** | $600 \times 500\text{ px}$ | **194 palms** | $2.57\text{ m}$ | $0.07$ | **$1.70$** | High shadow gradient and low-contrast illumination |
| **Sparse Boundary Edge** | $650 \times 470\text{ px}$ | **182 palms** | $2.67\text{ m}$ | $0.15$ | **$1.64$** | Transition margin with bare soil and non-palm cover |
| **Boundary Soil Transition** | $600 \times 500\text{ px}$ | **187 palms** | $2.62\text{ m}$ | $0.14$ | **$1.64$** | Peripheral boundary with uneven ground clutter |
| **Mendeley-Style Stressed Canopy** | $600 \times 500\text{ px}$ | **258 palms** | $2.52\text{ m}$ | $0.06$ | **$1.85$** | Chlorotic/yellowing foliage and nutrient deficit |
| **Multi-Scale / High Altitude** | $600 \times 600\text{ px}$ | **301 palms** | $2.54\text{ m}$ | $0.06$ | **$1.84$** | Higher flight altitude with smaller apparent crown diameter |
| **Wibowo Dense Interlocking** | $600 \times 500\text{ px}$ | **227 palms** | $2.55\text{ m}$ | $0.06$ | **$1.76$** | Maximum crown overlap and frond intermingling |
| **Full Survey Orthophoto** | $873 \times 1024\text{ px}$ | **641 palms** | $2.58\text{ m}$ | $0.09$ | **$1.73$** | Complete surveyed plantation block |

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/ryanradityatama/sawit-vision.git
cd sawit-vision
pip install -r requirements.txt
```

### 2. Run Stand Detection & Numbering (`count.py`)

Run palm tree detection with sliced window inference and automatic evaluation graphs:

```bash
# Ingest aerial image, slice at 640px, apply 20% overlap, and output to output_count/
python count.py --input count.jpg --tile 640 --overlap 0.20
```

**Output Artifacts Generated in `output_count/`:**
- `count_output.jpg`: Numbered audit preview ($1 \dots 641$).
- `count_detections.csv`: Roster with tree ID, center $(x, y)$, bounding box, and confidence.
- `block_summary.csv`: Stand inventory, observed density (trees/ha), and vacancy counts per block.
- `count_metrics_evaluation.png`: 4-panel analytical research dashboard.

### 3. Run Multi-Condition Generalization Benchmark

Benchmark the detector across all raw test datasets:

```bash
python scripts/benchmark_generalization.py --data-dir data/raw
```

### 4. Tiling & YOLO Fine-Tuning Pipeline

Prepare training tiles from large orthomosaics and fine-tune YOLO weights:

```bash
# 1. Slice orthomosaic and convert annotations to YOLO txt format
python scripts/make_tiles.py --image count.jpg --csv output_count/count_detections.csv --tile-size 640

# 2. Generate location-separated train/val/test splits
python scripts/make_splits.py --tiles-dir data/tiles --splits-dir data/splits

# 3. Train YOLO model with aerial nadir augmentations
python scripts/train.py --data configs/data.yaml --epochs 50 --batch 8 --imgsz 640

# 4. Export fine-tuned weights for zero-overhead CPU deployment
python scripts/export_onnx.py --weights models/best.pt --format torchscript
```

### 5. Open Dataset Catalog Tool

Query open academic and research datasets for palm detection and plantation mapping:

```bash
# List all 23 open datasets with descriptions and links
python scripts/download_data.py --list

# View download instructions for a specific dataset
python scripts/download_data.py --dataset mendeley_anom
```

---

## Repository Structure

```text
sawit-vision/
├── README.md                      # Project documentation and benchmarks
├── PRD.md                         # Product Requirements Document
├── ROADMAP.md                     # Engineering roadmap
├── implementation.md              # Technical architecture guide
├── requirements.txt               # Dependencies
├── count.py                       # Standalone palm counting and numbering program
├── configs/
│   ├── data.yaml                  # YOLO dataset paths
│   └── train.yaml                 # Training hyperparameters
├── models/
│   ├── best.pt                    # Fine-tuned PyTorch YOLO checkpoint
│   └── best.torchscript           # Compiled TorchScript model for CPU/edge
├── output_count/
│   ├── count_output.jpg           # Numbered audit preview image (1..N)
│   ├── count_detections.csv       # Tree coordinates and confidence roster
│   ├── block_summary.csv          # Management block density & vacancies
│   ├── count_metrics_evaluation.png # 4-panel analytical metrics dashboard
│   └── benchmark_comparison/      # Previews across all test conditions
├── data/
│   ├── raw/                       # Multi-condition benchmark image tiles
│   ├── tiles/                     # Sliced training images and YOLO txt labels
│   └── splits/                    # Spatial train/val/test split manifests
├── scripts/
│   ├── benchmark_generalization.py # Cross-dataset benchmark suite
│   ├── evaluate_count.py          # Metrics evaluation dashboard generator
│   ├── make_tiles.py              # Orthomosaic slicer and YOLO label converter
│   ├── make_splits.py             # Location-separated split generator
│   ├── train.py                   # YOLO fine-tuning script
│   ├── export_onnx.py             # Model compiler for edge deployment
│   └── download_data.py           # 23-dataset open catalog CLI
├── src/sawitvision/
│   ├── detector.py                # Deep learning & adaptive crown detectors
│   ├── infer.py                   # Sliced sliding-window inference & Greedy-NMM
│   ├── blocks.py                  # Block containment, regularity, vacancy detection
│   ├── viz.py                     # Visual numbering renderer and heatmaps
│   ├── geo.py                     # GeoJSON spatial vector export
│   └── metrics.py                 # Counting metrics suite (MAE, MAPE, bias, R²)
└── tests/                         # Unit tests (100% passing)
```

---

## Unit Test Suite

Run the automated test suite covering counting metrics, spatial geometry, block containment, and vacancy detection:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p "test_*.py"
# 10 tests passed (OK)
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
