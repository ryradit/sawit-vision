# Product Requirements Document (PRD)

## SawitVision: Palm Tree Detection and Counting from Aerial Imagery

| | |
|---|---|
| **Status** | Active / Baseline Implemented |
| **Owner** | Ryan Radityatama |
| **Last updated** | 27 Nov 2026 |
| **Type** | Independent Computer Vision & Geospatial System |

> **Scope Note:** SawitVision is an open-source technical system exploring palm tree detection, precision counting, and plantation mapping from aerial imagery. It relies exclusively on open public data, self-curated drone imagery, and custom code. Targets outlined here represent rigorous engineering benchmarks to validate across independent test environments.

---

## 1. Background

Agricultural estate managers and agronomists require reliable palm tree inventories for yield projection, input planning (fertilizers, treatments), replanting schedules, and carbon monitoring. Traditional census methods rely on manual ground walking or rough statistical sampling, which are slow, labor-intensive, and prone to significant spatial error across large estates.

High-resolution drone and aerial orthomosaics make comprehensive coverage feasible. However, automated computer vision counting introduces practical technical hurdles:
- **Canopy overlap:** Mature palm crowns interlock in dense commercial layouts.
- **Environmental variation:** Sunlight angle, deep cast shadows, cloud shadows, haze, and flight altitudes vary across flights.
- **Spectral and morphological distractors:** Immature palms, ground cover, understory weeds, and surrounding secondary jungle mimic target crown foliage.
- **Extreme image scale:** Drone orthomosaics regularly exceed 20,000 × 20,000 pixels, vastly exceeding deep learning model input dimensions.

---

## 2. Problem Statement

Plantation operators and agricultural intelligence platforms require an automated, repeatable pipeline that converts high-resolution aerial imagery into **verifiable tree counts, crown coordinates, and spatial density metrics** across defined management blocks, eliminating manual tallying.

---

## 3. Goals and Non-Goals

### Goals
1. Accurately detect individual oil palm trees in high-resolution aerial and drone imagery across dense, overlapping canopy conditions.
2. Produce reliable count statistics per image, management block (plot), and entire estate.
3. Generate standards-compliant, georeferenced spatial vectors (GeoJSON, Shapefile) compatible with open GIS platforms (e.g., QGIS).
4. Ingest massive orthomosaics through sliced windowed inference (tiling) with boundary deduplication and non-maximum merging.
5. Provide a deterministic numbering system (1 to N) and visual audit previews for manual validation and spot-checking.
6. Provide an asynchronous REST API and interactive interface for pipeline integration and non-technical review.
7. Ensure full reproducibility through versioned configurations, seed control, and location-split evaluation benchmarks.

### Non-Goals (v1)
- Foliar disease classification, nutrient deficiency diagnosis, or spectral stress profiling.
- Stand-level yield prediction or fresh fruit bunch (FFB) weight estimation.
- Low-resolution satellite imagery (e.g. 10 m Sentinel-2) where individual crowns cannot be resolved as distinct spatial objects.
- Multi-tenant enterprise SaaS infrastructure (billing, authentication, RBAC).
- Real-time onboard video streaming directly on flight controllers.

---

## 4. Personas and Stakeholders

| Persona | Core Workflows & Requirements |
|---|---|
| **Agronomist / Estate Manager** | Ingests drone surveys, verifies stand counts per planting block against planting records, spots stand vacancies, exports summary sheets without touching command-line scripts. |
| **Geospatial / GIS Analyst** | Loads vector detections directly into spatial software, overlays management boundaries, inspects density rasters, and tunes detection confidence thresholds. |
| **ML / Systems Engineer** | Interacts via clean REST endpoints, monitors inference performance, triggers batch pipelines, and analyzes evaluation reports on held-out flight locations. |

---

## 5. User Stories

1. **Total Count & Preview:** As an estate manager, I can upload an aerial photo or orthomosaic and receive an exact tree count along with a high-resolution preview featuring numbered crown bounding boxes.
2. **Block-Level Analytics:** As an agronomist, I can upload management boundary polygons and receive tree counts and stand density (trees per hectare) aggregated per block.
3. **Spatial Vector Interoperability:** As a GIS analyst, I can download detected crown centroids in standard GeoJSON format to overlay on estate basemaps.
4. **Sensitivity Tuning:** As an analyst, I can adjust confidence and intersection-over-union (IoU) thresholds to balance precision versus recall across variable lighting.
5. **Programmatic Pipeline:** As a software engineer, I can submit images to an API endpoint and poll for completion status, bounding box coordinates, and metadata.
6. **Uncertainty Audit:** As an agronomist, I can isolate low-confidence detections and export zoomed inspection crops to verify ambiguous vegetation before finalizing official counts.

---

## 6. Functional Requirements

Priority: **P0** = Minimum Viable Product, **P1** = Core Release, **P2** = Advanced Extensions.

| ID | Functional Requirement | Priority |
|---|---|---|
| FR-1 | Ingest standard raster formats: GeoTIFF, JPEG, PNG | P0 |
| FR-2 | Deep learning crown detection using trained PyTorch / YOLO architectures | P0 |
| FR-3 | Windowed sliced inference with configurable overlap and border non-maximum merging | P0 |
| FR-4 | Return total stand count and generate annotated preview with deterministically numbered bounding boxes (1..N) | P0 |
| FR-5 | Export detections to GeoJSON (projected map coordinates or normalized pixel space) and CSV | P0 |
| FR-6 | Configurable confidence, NMS, and slice dimension parameters | P0 |
| FR-7 | Asynchronous REST API: submit jobs, query status, retrieve results | P0 |
| FR-8 | Per-block spatial aggregation using polygon boundaries | P1 |
| FR-9 | Continuous spatial density heatmap generation (trees/ha grid) | P1 |
| FR-10 | Interactive web user interface for upload, interactive pan/zoom, and export | P1 |
| FR-11 | Standardized evaluation suite reporting detection metrics (mAP, F1) and counting error (MAE, MAPE, bias) | P0 |
| FR-12 | Uncertainty flagging and crop extraction for human-in-the-loop review | P1 |
| FR-13 | Webhook notification triggers upon asynchronous job completion | P2 |
| FR-14 | Multi-temporal orthomosaic comparison (stand change and tree loss over time) | P2 |
| FR-15 | Crop-and-classify module: extract detected crowns and categorize morphological/color attributes (e.g. canopy hue, maturity indicator) | P2 |

---

## 7. Non-Functional Requirements

| Metric Area | Target Requirement |
|---|---|
| **Accuracy** | Defined in Section 8 (Success Metrics). |
| **Throughput & Speed** | Process standard drone surveys efficiently via tiled GPU inference or optimized multi-core CPU execution. |
| **Memory Efficiency** | Windowed reading for high-resolution GeoTIFF files to prevent out-of-memory errors on massive orthomosaics. |
| **Fault Tolerance** | Idempotent job processing; structured error responses and logging. |
| **Reproducibility** | Explicit random seeds, deterministic sorting for numbering, versioned model weights and configs. |
| **Portability** | Multi-platform execution, containerized Docker deployments, and ONNX runtime support for edge/CPU inference. |
| **Data Integrity** | Strict reliance on open, public, and self-collected data; zero proprietary or restricted data commitments. |

---

## 8. Success Metrics

Evaluated strictly on held-out spatial test splits from distinct geographic locations and flight missions:

| Evaluation Metric | Target Benchmark |
|---|---|
| **Detection F1 (IoU = 0.5)** | ≥ 0.90 |
| **Detection mAP@0.5** | ≥ 0.90 |
| **Count Error (MAPE) per image/plot** | ≤ 5.0% |
| **Count Correlation ($R^2$)** | ≥ 0.98 |
| **Border Duplicate Rate** | < 1.0% |
| **Mean Absolute Error (MAE)** | Minimized, verified with near-zero systematic bias |

---

## 9. Assumptions and Dependencies

- Open aerial and drone orthomosaics provide sufficient spatial detail (GSD < 15 cm/pixel) to distinguish individual mature crowns.
- Framework stack: PyTorch, torchvision, open-source YOLO architectures, OpenCV, NumPy, Shapely, Rasterio/GeoPandas, FastAPI.
- Strict data compliance: All training and validation data sourced from open academic repositories, open aerial platforms, or self-annotated test sets.

---

## 10. Technical Risk Mitigation

| Identified Risk | Potential Impact | Mitigation Strategy |
|---|---|---|
| Canopy overlap in mature stands | Crown undercounting | Fine-tuned anchor-free detection heads, higher slice resolutions (e.g. 1024), and distance-aware non-maximum suppression |
| Geographic and sensor domain shift | Degraded accuracy on unseen sites | Geographic split validation, broad geometric and chromatic augmentations, explicit reporting across diverse lighting conditions |
| Boundary stitching artifacts | False duplicate counts | Sliced sliding windows with 20% spatial overlap and IoU/IOS deduplication |
| Disconnect between mAP and Count Error | High mAP yet biased stand counts | Evaluate explicit count metrics (MAE, MAPE, count bias) directly alongside standard object detection mAP |
