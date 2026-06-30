from typing import Any, Tuple, List, Union, Set, Optional
import numpy as np


def generate_pred_bulb_ann_layer(
        predicted_bulb_coords: List[Tuple[np.ndarray, np.ndarray]],
        voxelsize: np.ndarray,
        source_url: str,
        color: str = '#48b54c'
) -> dict[str, Any]:
    
    predicted_annotations = []
    for start_coords, end_coords in predicted_bulb_coords:
        pointA = (start_coords / (voxelsize * 10**6)).tolist()
        pointB = (end_coords / (voxelsize * 10**6)).tolist()

        predicted_annotations.append({
                'pointA': pointA,
                'pointB': pointB,
                'type': 'line',
                'id': ''
        })

    predicted_layer = {
        'type': 'annotation',
        'source': source_url,
        'tool': 'annotateLine',
        'tab': 'annotations',
        'annotationColor': color,
        'annotations': predicted_annotations,
        'name': 'Predicted bulbs'
    }

    return predicted_layer


def generate_skel_ann_layer(
        skel: Any,
        voxelsize: np.ndarray,
        source_url: str,
        color: str = '#ff0000'
) -> dict[str, Any]:
    
    skel_graph = skel._igraph()
    edges = skel_graph.get_edgelist()
    full_skel_ann = []

    for n1, n2 in edges:
        start_coords = skel.nodes[n1]
        end_coords = skel.nodes[n2]

        pointA = (start_coords / (voxelsize * 10**6)).tolist()
        pointB = (end_coords / (voxelsize * 10**6)).tolist()

        full_skel_ann.append({
                'pointA': pointA,
                'pointB': pointB,
                'type': 'line',
                'id': ''
            })

    skel_layer = {
        'type': 'annotation',
        'source': source_url,
        'tool': 'annotateLine',
        'tab': 'annotations',
        'annotationColor': color,
        'annotations': full_skel_ann,
        'name': 'skeleton'
    }

    return skel_layer


def generate_radii_ann_layer(
        skel: Any,
        voxelsize: np.ndarray,
        source_url: str,
        color: str = '#0000ff'
) -> dict[str, Any]:
    
    radii = skel.radii['calibrated']
    radii_annotations = []

    for i, coords in enumerate(skel.nodes):

        center = (coords / (voxelsize * 10**6)).tolist()
        ellipsoid_radii = (np.array([radii[i], radii[i], radii[i]]) / (voxelsize * 10**6)).tolist()

        radii_annotations.append({
            'center': center,
            'radii': ellipsoid_radii,
            'type': 'ellipsoid',
            'id': ''
        })

    radii_layer = {
        'type': 'annotation',
        'source': source_url,
        'tool': 'annotateSphere',
        'tab': 'annotations',
        'annotationColor': color,
        'annotations': radii_annotations,
        'name': 'radii_annotations'
    }

    return radii_layer

