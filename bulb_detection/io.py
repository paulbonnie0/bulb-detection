from pathlib import Path
from typing import Any, Tuple, List
import pickle
import json
import numpy as np


def load_skeleton(filepath: str | Path) -> Any:
    """Loads a skeliner skeleton from a .pkl file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def load_mesh(filepath: str | Path) -> Any:
    """Loads a cell mesh from a .pkl file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def parse_annotations_json(filepath: str | Path) -> dict[str, Any]:
    """Loads a neuroglancer state json."""
    with open(filepath, 'rb') as f:
        return json.load(f)


def get_layer_voxelsize(ann: dict[str, Any], layer_id: int) -> np.ndarray:
    """Returns the [x, y, z] voxel to meter conversion for a specific layer."""
    voxelsize = np.array([ann['layers'][layer_id]['source']['transform']['outputDimensions'][plane][0] for plane in ['x','y','z']])
    return voxelsize


def get_annotated_bulb_coords(
        ann: dict[str, Any],
        layer_id: int,
        voxelsize: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the the start and endpoint coordinates for all bulbs in the annotation."""
    annotated_bulb_starts = []
    annotated_bulb_ends = []
    for bulb in ann['layers'][layer_id]['annotations']:
        annotated_bulb_starts.append(bulb['pointA'])
        annotated_bulb_ends.append(bulb['pointB'])
    # convert meter to micrometer
    annotated_bulb_starts = np.array(annotated_bulb_starts) * voxelsize*10**6
    annotated_bulb_ends  =np.array(annotated_bulb_ends) * voxelsize*10**6
    
    return annotated_bulb_starts, annotated_bulb_ends


def get_endpoint_coords(
        ann: dict[str, Any],
        layer_id: int,
        voxelsize: np.ndarray
) -> List[np.ndarray]:
    """Returns the coordinates for all endpoints in an annotation endpoint layer."""
    endpoints = []
    for endpoint_ann in ann['layers'][layer_id]['annotations']:
        endpoints.append(np.array(endpoint_ann['point']) * voxelsize * 10**6)
        
    return endpoints
