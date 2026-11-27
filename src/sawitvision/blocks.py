"""Management block spatial containment, planting regularity, and stand vacancy detection."""

import json
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy.spatial import Delaunay, KDTree


class ManagementBlock:
    """Represents a plantation management block defined by a polygon."""

    def __init__(
        self,
        block_id: str,
        name: str,
        polygon_coords: List[Tuple[float, float]],
        target_density_per_ha: float = 136.0,
    ):
        self.block_id = block_id
        self.name = name
        self.polygon_coords = np.array(polygon_coords, dtype=np.float32)
        self.target_density_per_ha = target_density_per_ha

    def contains_point(self, x: float, y: float) -> bool:
        """Check if point (x, y) is inside or on the boundary of the polygon."""
        pt = (float(x), float(y))
        res = cv2.pointPolygonTest(self.polygon_coords, pt, False)
        return res >= 0

    def compute_area_px(self) -> float:
        """Calculate polygon area in square pixels using Green's theorem."""
        return float(abs(cv2.contourArea(self.polygon_coords)))

    def compute_area_hectares(self, gsd_m: float = 0.08) -> float:
        """Calculate polygon area in hectares based on Ground Sampling Distance (GSD in meters/pixel).
        
        1 hectare = 10,000 square meters.
        """
        area_px = self.compute_area_px()
        area_m2 = area_px * (gsd_m**2)
        return float(area_m2 / 10000.0)


def assign_trees_to_blocks(
    detections: List[Dict[str, Any]],
    blocks: List[ManagementBlock],
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Assign detected tree centroids to their containing management block.
    
    Returns:
        tuple: (block_tree_map, unassigned_trees)
    """
    block_map: Dict[str, List[Dict[str, Any]]] = {b.block_id: [] for b in blocks}
    unassigned: List[Dict[str, Any]] = []

    for det in detections:
        cx, cy = det["center_x"], det["center_y"]
        assigned_block = None

        for b in blocks:
            if b.contains_point(cx, cy):
                assigned_block = b.block_id
                det["block_id"] = assigned_block
                block_map[assigned_block].append(det)
                break

        if assigned_block is None:
            det["block_id"] = "Unassigned"
            unassigned.append(det)

    return block_map, unassigned


def calculate_planting_regularity(
    tree_points: np.ndarray,
    gsd_m: float = 0.08,
) -> Dict[str, float]:
    """Evaluate planting grid regularity using Nearest-Neighbor Distance (NND) and Clark-Evans Index."""
    if len(tree_points) < 4:
        return {
            "mean_spacing_m": 0.0,
            "std_spacing_m": 0.0,
            "cv_spacing": 0.0,
            "clark_evans_r": 0.0,
            "regularity_score": 0.0,
        }

    kdtree = KDTree(tree_points)
    # Query distance to nearest neighbor (k=2: 1st is point itself, 2nd is nearest neighbor)
    distances, _ = kdtree.query(tree_points, k=2)
    nnd_px = distances[:, 1]
    nnd_m = nnd_px * gsd_m

    mean_spacing_m = float(np.mean(nnd_m))
    std_spacing_m = float(np.std(nnd_m))
    cv = float(std_spacing_m / max(1e-5, mean_spacing_m))

    # Clark-Evans Aggregation Index: R = r_obs / r_exp
    # r_exp = 1 / (2 * sqrt(density))
    # Estimate bounding box area
    min_x, min_y = np.min(tree_points, axis=0)
    max_x, max_y = np.max(tree_points, axis=0)
    box_area_m2 = max(1e-3, (max_x - min_x) * (max_y - min_y) * (gsd_m**2))
    density_per_m2 = len(tree_points) / box_area_m2
    r_exp_m = 1.0 / (2.0 * np.sqrt(max(1e-6, density_per_m2)))
    clark_evans_r = float(mean_spacing_m / max(1e-5, r_exp_m))

    # Regularity Score (0 to 100): perfectly regular commercial triangular lattices have R ~ 1.8-2.1 and low CV (< 0.15)
    # Scaled where R >= 1.5 and CV <= 0.20 scores > 85%
    regularity_score = float(np.clip((1.0 - cv) * (min(clark_evans_r, 2.0) / 2.0) * 100.0, 0.0, 100.0))

    return {
        "mean_spacing_m": round(mean_spacing_m, 2),
        "std_spacing_m": round(std_spacing_m, 2),
        "cv_spacing": round(cv, 3),
        "clark_evans_r": round(clark_evans_r, 2),
        "regularity_score": round(regularity_score, 1),
    }


def detect_stand_vacancies(
    tree_points: np.ndarray,
    block_polygon: np.ndarray,
    nominal_spacing_px: Optional[float] = None,
    gsd_m: float = 0.08,
) -> List[Dict[str, Any]]:
    """Locate missing palm tree positions (stand vacancies) inside the management block using Delaunay lattice analysis.
    
    Identifies vacant grid positions where an expected palm is missing due to mortality, disease, or planting gaps.
    """
    if len(tree_points) < 8:
        return []

    kdtree = KDTree(tree_points)

    if nominal_spacing_px is None:
        distances, _ = kdtree.query(tree_points, k=2)
        nominal_spacing_px = float(np.median(distances[:, 1]))

    # Minimum margin from polygon boundary to avoid boundary edge false positives
    boundary_margin = nominal_spacing_px * 0.85

    tri = Delaunay(tree_points)
    candidate_vacancies = []

    for simplex in tri.simplices:
        pts = tree_points[simplex]
        # Calculate triangle centroid
        centroid = np.mean(pts, axis=0)
        cx, cy = float(centroid[0]), float(centroid[1])

        # 1. Must be strictly inside block polygon with safe boundary margin
        dist_to_boundary = cv2.pointPolygonTest(block_polygon, (cx, cy), True)
        if dist_to_boundary < boundary_margin:
            continue

        # 2. Distance from centroid to nearest detected tree
        dist_to_nearest, _ = kdtree.query((cx, cy))

        # In an ideal equilateral triangle with side L, distance from centroid to vertex is L / sqrt(3) ~ 0.577 * L
        # If there is a missing tree at this location, distance to nearest tree is ~ 0.65 * L to 1.4 * L
        if dist_to_nearest >= nominal_spacing_px * 0.62:
            candidate_vacancies.append((cx, cy, dist_to_nearest))

    # Cluster / Deduplicate candidate vacancies that are close together
    filtered_vacancies = []
    min_separation = nominal_spacing_px * 0.70

    for cx, cy, dist_near in candidate_vacancies:
        is_duplicate = False
        for existing in filtered_vacancies:
            ex_x, ex_y = existing["center_x"], existing["center_y"]
            sep = np.hypot(cx - ex_x, cy - ex_y)
            if sep < min_separation:
                is_duplicate = True
                break

        if not is_duplicate:
            conf = float(np.clip(dist_near / nominal_spacing_px, 0.5, 1.0))
            filtered_vacancies.append(
                {
                    "center_x": round(cx, 1),
                    "center_y": round(cy, 1),
                    "dist_nearest_m": round(float(dist_near * gsd_m), 2),
                    "confidence": round(conf, 3),
                }
            )

    # Number vacancies sequentially
    for idx, v in enumerate(filtered_vacancies, start=1):
        v["vacancy_id"] = f"V-{idx:03d}"

    return filtered_vacancies


def compute_block_summary(
    block: ManagementBlock,
    trees_in_block: List[Dict[str, Any]],
    vacancies: List[Dict[str, Any]],
    gsd_m: float = 0.08,
) -> Dict[str, Any]:
    """Generate detailed agronomic stand statistics for a management block."""
    actual_count = len(trees_in_block)
    area_ha = block.compute_area_hectares(gsd_m)
    stand_density_ha = actual_count / max(1e-4, area_ha)

    target_count = int(round(area_ha * block.target_density_per_ha))
    vacancy_count = len(vacancies)
    effective_stand = actual_count
    stand_deficit = max(0, target_count - actual_count)
    completion_rate = (actual_count / max(1, target_count)) * 100.0

    tree_pts = np.array([[t["center_x"], t["center_y"]] for t in trees_in_block], dtype=np.float32)
    regularity = calculate_planting_regularity(tree_pts, gsd_m)

    return {
        "block_id": block.block_id,
        "name": block.name,
        "area_ha": round(area_ha, 3),
        "actual_trees": actual_count,
        "target_trees": target_count,
        "observed_density_ha": round(stand_density_ha, 1),
        "target_density_ha": block.target_density_per_ha,
        "completion_rate_pct": round(completion_rate, 1),
        "detected_vacancies": vacancy_count,
        "stand_deficit": stand_deficit,
        "mean_spacing_m": regularity["mean_spacing_m"],
        "spacing_cv": regularity["cv_spacing"],
        "regularity_score": regularity["regularity_score"],
    }
