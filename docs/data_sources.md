# Data Sources & Benchmark Catalog

Comprehensive open dataset catalog for oil palm tree detection, precision counting, health anomaly identification, and regional plantation mapping.

> **Provenance Standard:** All datasets listed below are strictly open research or publicly accessible datasets (CC-BY, Open Academic, or Open Data). No raw imagery is committed to git; use the references below or run `python scripts/download_data.py --list` to inspect and download.

---

## 1. Direct Oil Palm Drone & Aerial Detection Datasets

| # | Dataset Name | Platform / Source | Sensor / Altitude | Target Annotations | Primary Research Use | Reference / Link |
|---|---|---|---|---|---|---|
| 1 | **Ganoderma-Affected Oil Palm Trees** | Mendeley Data | UAV RGB (0.04 m/px GSD) | 9,400 crown bounding boxes in COCO format | Differentiates healthy palms vs. Ganoderma basal stem rot infected crowns | [DOI: 10.17632/m4pmx9w7b8.1](https://data.mendeley.com/) |
| 2 | **Oil Palm Anomaly Identification** (Dominguez Meza et al.) | Mendeley Data | UAV Drone (17m–30m) | Bounding boxes: `PalmSan` (healthy) vs. `PalmAnom` (stressed/diseased) | Crown localization and health classification under variable altitudes | [DOI: 10.17632/nh7d23dgnw.1](https://data.mendeley.com/datasets/nh7d23dgnw/1) |
| 3 | **PALMS Dataset** (Ecuador Rainforest Consortium) | arXiv:2502.13023 | UAV Orthomosaics across 21 sites | 8,830 bounding boxes + 5,026 crown center points | Crown detection in multi-scale tropical canopies; PRISM pipeline benchmark | [arXiv:2502.13023](https://arxiv.org/html/2502.13023v2) |
| 4 | **OilPalm_Dataset.id** | Mendeley Data | UAV Aerials (Kalimantan Tengah, Indonesia) | Regional plantation orthomosaics & tiles | Regional transfer learning on Indonesian peat and mineral soil estates | [Mendeley Data](https://data.mendeley.com/) |
| 5 | **Palm Tree Detection & Counting** | Kaggle Open Data | Overhead UAV / Drone tiles | Bounding boxes formatted for YOLO & Faster R-CNN | Stand inventory counting, density estimation, and validation splits | [Kaggle Palm Datasets](https://www.kaggle.com/) |
| 6 | **Wibowo et al. UAV Oil Palm Benchmark** | MDPI Drones (2022) | High-res drone (5 cm/px GSD) | Evaluated on sparse, dense, and interlocking canopies | Benchmark for small-object detection under dense overlapping fronds | [MDPI Drones 2022](https://www.mdpi.com/2504-2289/6/3/89) |

---

## 2. Palm Family Benchmarks (Date, Coconut & Areca Palms)

Useful for pretraining feature backbones on radial palm frond geometry before fine-tuning on oil palm:

| # | Dataset Name | Platform / Source | Sensor / Region | Annotations | Primary Research Use |
|---|---|---|---|---|---|
| 7 | **Dataset Ninja Palm Trees** | Dataset Ninja | UAV Pro (Kharj, Saudi Arabia) | ~10,000 annotated instances with georeferences | Large-scale pretraining on regular plantation grids |
| 8 | **UAE High-Resolution UAV Palm Dataset** | Copernicus / UAE | UAV Aerials (Dubai, Al-Ain, Khorfakkan) | Bounding boxes & pixel masks (YOLO, DETR) | Pretraining crown feature extractors under harsh sunlight and high contrast |
| 9 | **Aerial Images of Palm Trees** | Kaggle | UAV Drone surveys | Bounding boxes across farm layouts | Data augmentation and cross-sensor robustness testing |
| 10 | **Ghana Coconut Palm Drone Dataset** | Research Archive | Drone surveys (10m–85m altitude) | High-density coconut crown annotations | Multi-altitude scale invariance and synthetic-to-real domain adaptation |
| 11 | **Al-Saad Palm Tree Benchmark** | Academic Open Benchmark | Very-high-resolution aerial imagery | Crown center coordinates and bounding boxes | Spatial point pattern analysis and crown centroid regression |
| 12 | **Multiscale Date Palm Segmentation** | MDPI Remote Sensing | UAV Multispectral & RGB | High-resolution polygon crown masks | Semantic segmentation (SegFormer, Swin Transformer) vs bounding boxes |

---

## 3. Macro Plantation Delineation & Satellite Datasets (0.3m–10m)

Used for macro estate boundary delineation (Module A) to mask agricultural parcels before drone-based counting:

| # | Dataset Name | Platform / Source | Sensor / Resolution | Target Classes | Primary Research Use |
|---|---|---|---|---|---|
| 13 | **Indonesian Palm Oil Reference Dataset** (Wafiq et al.) | Zenodo / Lacuna Fund | 0.3m Satellite raster + Sentinel-2 10m | Young palm, mature palm, cacao, rubber, built-up | Expert ground truth for plantation segmentation across Sumatra and Kalimantan |
| 14 | **Global Oil Palm Extent & Planting Year** (Descals et al.) | Zenodo / Earth Engine | Sentinel-1/2 10m + Landsat 30m | Industrial palm, smallholder palm, planting year (1990–2021) | Continental-scale masking and plantation age profiling |
| 15 | **Indonesian Plantation Expansion** (Gaveau et al.) | Zenodo Open Data | Multi-temporal Landsat / Sentinel | Annual industrial vs. smallholder boundaries (2001–present) | Historical expansion analysis and concession boundary validation |
| 16 | **Southeast Asia Sentinel-2 Palm Maps (2022–2024)** | Zenodo (DOI: 10.5281/zenodo.17768444) | Sentinel-2 (10m Surface Reflectance) | Annual country-scale oil palm binary masks | Temporal change detection and baseline verification |
| 17 | **ForestNet Dataset** | Stanford ML / HuggingFace | Landsat / Sentinel multi-band | Deforestation drivers (oil palm, logging, pasture) | Differentiating commercial plantations from native forest clearing |

---

## 4. Precision Agroforestry & Individual Tree Crown (ITC) Benchmarks

Broad individual tree crown (ITC) benchmarks for testing sliding-window tiling and crown clustering:

| # | Dataset Name | Platform / Source | Sensor / Platform | Annotations | Primary Research Use |
|---|---|---|---|---|---|
| 18 | **NEON Tree Crowns Dataset** | Zenodo / Weecology | Airborne RGB + LiDAR (USA-wide) | 100M+ individual tree crowns, height & bounding boxes | Benchmark for large-scale spatial tiling and tree crown clustering |
| 19 | **DeepForest Benchmark** | Zenodo / GitHub | UAV & Airborne RGB orthomosaics | Standardized individual crown bounding boxes | Pretrained baseline comparison against standard forestry detectors |
| 20 | **Detectree2 Tropical UAV Benchmark** | Zenodo | UAV RGB over Danum & Sepilok (Malaysia) | Closed-canopy tropical tree polygon masks | Testing boundary deduplication in dense Southeast Asian rain forests |
| 21 | **Campaneta Orange Orchard UAV Dataset** | Zenodo | UAV RGB + Multispectral (Mavic 3M) | Fruit and canopy bounding boxes in commercial grids | Benchmarking grid regularity algorithms in row-planted orchards |
| 22 | **TreeAI Global Initiative Database** | Zenodo | High-resolution aerials (Global) | COCO-format bounding boxes & segmentation masks | Cross-species canopy feature extraction |
| 23 | **OpenAerialMap Southeast Asia Drone Orthomosaics** | OpenAerialMap (OAM) | Cloud-Optimized GeoTIFFs (< 10 cm GSD) | Unannotated full plantation orthomosaics | Testing large-scale windowed sliding inference on real gigapixel rasters |
