#!/usr/bin/env python3
"""Dataset acquisition and catalog utility for palm tree detection and plantation mapping."""

import argparse
import os
import sys

DATASET_CATALOG = {
    # 1. Direct Oil Palm Drone Datasets
    "ganoderma_uav": {
        "title": "UAV RGB Image Dataset for Object Detection of Ganoderma-Affected Oil Palm Trees",
        "platform": "Mendeley Data",
        "sensor": "UAV RGB Drone (0.04 m/px GSD)",
        "classes": ["Healthy Palm (Survey-Negative)", "Ganoderma-Infected Palm"],
        "annotation_format": "9,400 Bounding Boxes (COCO JSON / 640x640 tiles)",
        "license": "CC BY 4.0",
        "url": "https://data.mendeley.com/",
        "description": "High-resolution drone orthomosaics with 9,400 crown boxes explicitly split into disjoint geographic blocks to prevent leakage.",
    },
    "mendeley_anom": {
        "title": "Oil Palm Tree Detection for Anomaly Identification",
        "platform": "Mendeley Data (DOI: 10.17632/nh7d23dgnw.1)",
        "sensor": "UAV Drone (17m to 30m altitude)",
        "classes": ["PalmSan (Healthy)", "PalmAnom (Stressed / Diseased)"],
        "annotation_format": "Bounding Boxes (Pascal VOC / YOLO)",
        "license": "CC BY 4.0",
        "url": "https://data.mendeley.com/datasets/nh7d23dgnw/1",
        "description": "Drone imagery of oil palms labeled into healthy and anomalous crowns for stress assessment and detection.",
    },
    "palms_ecuador": {
        "title": "PALMS Dataset (PAlm Localization in Multi-Scale)",
        "platform": "arXiv:2502.13023 / Research Open Repository",
        "sensor": "UAV Multi-Scale Orthomosaics (21 diverse tropical sites)",
        "classes": ["Palm Crown"],
        "annotation_format": "8,830 Bounding Boxes & 5,026 Crown Center Points",
        "license": "Open Academic",
        "url": "https://arxiv.org/html/2502.13023v2",
        "description": "Extensive multi-site UAV orthomosaics across tropical canopies. Benchmark for large-image tiling and PRISM framework.",
    },
    "oilpalm_id": {
        "title": "OilPalm_Dataset.id (Central Kalimantan, Indonesia)",
        "platform": "Mendeley Data",
        "sensor": "UAV Aerial Drone Orthophotos",
        "classes": ["Oil Palm Crown"],
        "annotation_format": "High-resolution regional tiles",
        "license": "Open Research",
        "url": "https://data.mendeley.com/",
        "description": "Imagery collected across Central Kalimantan peatland and mineral oil palm plantations.",
    },
    "kaggle_palm_count": {
        "title": "Palm Tree Detection and Counting Dataset",
        "platform": "Kaggle Open Data",
        "sensor": "Drone Aerial Orthomosaics",
        "classes": ["Palm Tree"],
        "annotation_format": "YOLO / Faster R-CNN Bounding Boxes",
        "license": "Open Community",
        "url": "https://www.kaggle.com/",
        "description": "Overhead drone tiles curated for automated tree inventory counting and density estimation.",
    },
    "wibowo_benchmark": {
        "title": "Wibowo et al. UAV Oil Palm Benchmark (2022)",
        "platform": "MDPI Drones (DOI: 10.3390/drones6030089)",
        "sensor": "UAV Drone (5 cm/px GSD)",
        "classes": ["Sparse, Dense, and Overlapping Palms"],
        "annotation_format": "Crown Bounding Boxes & Spatial Points",
        "license": "CC BY 4.0",
        "url": "https://www.mdpi.com/2504-2289/6/3/89",
        "description": "Benchmark testing object detectors under sparse, dense, and interlocking canopy layouts.",
    },

    # 2. Palm Family Benchmarks (Date, Coconut, Areca)
    "saudi_date_palms": {
        "title": "Dataset Ninja Palm Trees Dataset (Kharj, Saudi Arabia)",
        "platform": "Dataset Ninja",
        "sensor": "UAV Pro Drone Aerials",
        "classes": ["Date Palm Tree"],
        "annotation_format": "~10,000 Annotated Instances with Geotags",
        "license": "Open Dataset",
        "url": "https://datasetninja.com/",
        "description": "Large-scale drone dataset over commercial palm farms for spatial point pattern and grid analysis.",
    },
    "uae_uav_palms": {
        "title": "UAE High-Resolution UAV Palm Dataset",
        "platform": "Copernicus / UAE Open Research",
        "sensor": "UAV Aerial Orthomosaics",
        "classes": ["Palm Crown", "Foliage Mask"],
        "annotation_format": "YOLO / DETR Bounding Boxes and Masks",
        "license": "Open Academic",
        "url": "https://copernicus.org/",
        "description": "UAV imagery across arid and irrigated estates for crown extraction under high solar contrast.",
    },
    "ghana_coconut_uav": {
        "title": "Ghana Coconut Palm UAV & Synthetic Dataset",
        "platform": "The Moonlight / Research Archive",
        "sensor": "UAV Drone (10m to 85m altitude)",
        "classes": ["Coconut Palm"],
        "annotation_format": "Multi-altitude YOLO bounding boxes",
        "license": "Open Research",
        "url": "https://themoonlight.io/",
        "description": "Drone imagery across variable altitudes testing scale invariance and synthetic augmentation.",
    },

    # 3. Macro Plantation & Satellite Datasets
    "indonesia_zenodo": {
        "title": "Open Land Use Reference Dataset for Palm Oil Landscapes in Indonesia",
        "platform": "Zenodo (DOI: 10.5281/zenodo.15618532)",
        "sensor": "High-Resolution Satellite (0.3m) with Sentinel-2 Composites (10m)",
        "classes": ["Young Palm", "Mature Palm", "Cacao", "Rubber", "Built-Up"],
        "annotation_format": "Wall-to-wall Polygon Label Rasters & GeoJSON",
        "license": "CC BY 4.0",
        "url": "https://zenodo.org/records/15618532",
        "description": "Expert ground truth across Indonesian landscapes for macro estate boundary segmentation.",
    },
    "global_oilpalm_map": {
        "title": "Global Oil Palm Extent & Planting Year (1990-2021)",
        "platform": "Zenodo (DOI: 10.5281/zenodo.3884601)",
        "sensor": "Sentinel-1/2 10m & Landsat 30m Time Series",
        "classes": ["Industrial Closed-Canopy Palm", "Smallholder Palm", "Planting Year"],
        "annotation_format": "10m Classified GeoTIFF Rasters & Validation Points",
        "license": "CC BY 4.0",
        "url": "https://zenodo.org/records/3884601",
        "description": "Global baseline oil palm inventory map for macro estate masking.",
    },
    "sea_sentinel_maps": {
        "title": "Oil Palm Maps for Malaysia and Indonesia (2022-2024)",
        "platform": "Zenodo (DOI: 10.5281/zenodo.17768444)",
        "sensor": "Sentinel-2 Multispectral Surface Reflectance (10m)",
        "classes": ["Oil Palm Plantation Extent"],
        "annotation_format": "Multi-temporal GeoTIFF binary masks",
        "license": "CC BY 4.0",
        "url": "https://doi.org/10.5281/zenodo.17768444",
        "description": "Country-scale annual maps for temporal change monitoring and baseline verification.",
    },
    "forestnet": {
        "title": "ForestNet Deforestation Drivers Dataset",
        "platform": "Stanford ML / HuggingFace",
        "sensor": "Landsat 8 & Sentinel-2 Satellite Composites",
        "classes": ["Oil Palm Plantation", "Timber Plantation", "Grassland", "Smallholder"],
        "annotation_format": "Satellite patches with driver labels",
        "license": "Open Academic",
        "url": "https://huggingface.co/",
        "description": "Satellite benchmark for differentiating oil palm plantations from other land covers.",
    },

    # 4. Precision Agroforestry & Individual Tree Crown (ITC) Datasets
    "neon_tree_crowns": {
        "title": "NEON Tree Crowns Dataset",
        "platform": "Zenodo / Weecology",
        "sensor": "Airborne RGB + LiDAR",
        "classes": ["Individual Tree Crowns"],
        "annotation_format": "100M+ Bounding Boxes and Height Attributes",
        "license": "CC BY 4.0",
        "url": "https://zenodo.org/",
        "description": "Massive collection of tree crowns across diverse forest sites for sliding window benchmarking.",
    },
    "deepforest": {
        "title": "DeepForest Benchmark Dataset",
        "platform": "Zenodo / GitHub",
        "sensor": "UAV & Airborne RGB Orthomosaics",
        "classes": ["Tree Crown"],
        "annotation_format": "Standardized Bounding Boxes",
        "license": "MIT",
        "url": "https://github.com/weecology/DeepForest",
        "description": "Airborne benchmark for individual tree crown delineation.",
    },
    "detectree2_tropical": {
        "title": "Detectree2 Tropical UAV Benchmark (Danum & Sepilok, Malaysia)",
        "platform": "Zenodo",
        "sensor": "UAV RGB Drone Orthomosaics",
        "classes": ["Tropical Forest Tree Crowns"],
        "annotation_format": "Polygon Crown Masks in COCO format",
        "license": "CC BY 4.0",
        "url": "https://zenodo.org/",
        "description": "Dense closed-canopy tropical forest crowns in Malaysia for instance segmentation.",
    },
    "campaneta_orchard": {
        "title": "Campaneta Citrus Orchard UAV Dataset",
        "platform": "Zenodo",
        "sensor": "UAV RGB + 4-Band Multispectral (Mavic 3M)",
        "classes": ["Tree Canopy", "Individual Fruit"],
        "annotation_format": "Canopy & Fruit Bounding Boxes",
        "license": "CC BY 4.0",
        "url": "https://zenodo.org/",
        "description": "Synchronized high-density UAV orchard imagery for spacing and yield estimation.",
    },
    "treeai_global": {
        "title": "TreeAI Global Tree Species Identification Database",
        "platform": "Zenodo",
        "sensor": "High-Resolution Aerial & UAV Imagery",
        "classes": ["Multi-Species Tree Canopies"],
        "annotation_format": "COCO Bounding Boxes & Polygon Masks",
        "license": "CC BY 4.0",
        "url": "https://zenodo.org/",
        "description": "Global database for cross-species canopy feature extraction and classification.",
    },
    "openaerialmap_uav": {
        "title": "OpenAerialMap UAV Orthomosaics (Southeast Asia)",
        "platform": "OpenAerialMap",
        "sensor": "High-Resolution Drone RGB (< 10 cm/px GSD)",
        "classes": ["Unannotated High-Res Agricultural Rasters"],
        "annotation_format": "Cloud-Optimized GeoTIFF (COG)",
        "license": "CC BY 4.0",
        "url": "https://openaerialmap.org/",
        "description": "Free, open-license gigapixel drone orthomosaics across Sumatra and Kalimantan.",
    },
}


def print_catalog():
    print("\n" + "=" * 92)
    print("SAWITVISION OPEN DATASET CATALOG (20+ VERIFIED BENCHMARKS)")
    print("=" * 92)
    for idx, (key, info) in enumerate(DATASET_CATALOG.items(), start=1):
        print(f"\n[{idx:02d}] {info['title']}")
        print(f"     Key           : {key}")
        print(f"     Platform      : {info['platform']} (License: {info['license']})")
        print(f"     Sensor / Res  : {info['sensor']}")
        print(f"     Annotations   : {info['annotation_format']}")
        print(f"     Access Link   : {info['url']}")
        print(f"     Description   : {info['description']}")
    print("\n" + "=" * 92 + "\n")


def print_download_instructions(dataset_key: str):
    if dataset_key not in DATASET_CATALOG:
        print(f"Error: Unknown dataset key '{dataset_key}'. Use --list to view all available keys.")
        return

    info = DATASET_CATALOG[dataset_key]
    dest_dir = os.path.join("data", "raw", dataset_key)
    print(f"\nDownload Instructions for: {info['title']}")
    print(f"Target Directory: {dest_dir}/")
    print("-" * 75)
    print(f"1. Open dataset repository in browser: {info['url']}")
    print(f"2. Download archive files (.zip, .tar, .tif).")
    print(f"3. Extract image and annotation files into: {dest_dir}/")
    print(f"4. Run: python scripts/benchmark_generalization.py --data-dir {dest_dir}")
    print("-" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Query open palm tree and agroforestry datasets.")
    parser.add_argument("--list", action="store_true", default=True, help="List all 20+ open datasets.")
    parser.add_argument("--dataset", "-d", choices=list(DATASET_CATALOG.keys()), help="Dataset key for download instructions.")

    args = parser.parse_args()

    if args.dataset:
        print_download_instructions(args.dataset)
    else:
        print_catalog()


if __name__ == "__main__":
    main()
