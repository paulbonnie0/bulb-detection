from typing import Any, Tuple, List, Union
import copy
import numpy as np
from scipy.spatial import KDTree


def coords_to_nodes(skel: Any, query_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the clostest node_ids and distances to those nodes for the query coords."""
    tree = KDTree(skel.nodes)
    distances, node_ids = tree.query(query_coords)
    return node_ids, distances # type: ignore[return-value]


def compute_geodesic_distances(skel: Any, soma_node_id: int = 0) -> np.ndarray:
    """Returns the geodesic distance (in μm) from the soma for all skeleton nodes."""
    skel_graph = skel._igraph()
    edges = np.array(skel_graph.get_edgelist())

    edge_weights = []
    coords_n1 = skel.nodes[edges[:, 0]]
    coords_n2 = skel.nodes[edges[:, 1]]
    edge_weights = np.linalg.norm(coords_n1 - coords_n2, axis=1)

    distances_array = np.array(skel_graph.distances(source=soma_node_id, weights=edge_weights)[0])

    return distances_array


def prune_skeleton_from_node_ids(skel: Any, keep_node_ids: np.ndarray) -> Any:
    """Prunes all nodes from the skeleton that are not on the shortest paths between the soma
    and the keep_nodes."""
    pruned_skel = copy.deepcopy(skel)

    skel_graph = pruned_skel._igraph()
    keep_nodes = []

    paths = skel_graph.get_shortest_paths(0, to=keep_node_ids)
    for path in paths:
        keep_nodes.extend(path)

    keep_nodes_set = set(keep_nodes)
    total_nodes_set = set(range(len(pruned_skel.nodes)))
    remove_nodes_set = total_nodes_set - keep_nodes_set

    pruned_skel.prune(kind="nodes", nodes=remove_nodes_set)
    
    return pruned_skel


def get_bulb_clusters(
        skel: Any,
        start_node_ids: np.ndarray,
        end_node_ids: np.ndarray
) -> List[List[int]]:
    """
    Takes bulb start_node_ids and end_node_ids and
    returns the bulb clusters (list of lists where each list is a bulb/cluster).
    """
    skel_graph = skel._igraph()

    bulb_clusters = []
    for start_node_id, end_node_id in zip(start_node_ids, end_node_ids):

        bulb_path = skel_graph.get_shortest_paths(start_node_id, to=end_node_id)[0]
        bulb_clusters.append(bulb_path)

    return bulb_clusters


def filter_bulb_coords_by_endpoints(
        skel: Any,
        starts_coords: np.ndarray,
        ends_coords: np.ndarray,
        endpoint_nodes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filters the global bulb starts_coords/ends_coords to only contain
    bulbs that are on the path to the endpoint_nodes.
    """
    skel_graph = skel._igraph()
    endpoint_path_nodes = set()
    for node in endpoint_nodes:
        path = skel_graph.get_shortest_paths(0, to=node)[0]
        endpoint_path_nodes.update(path)

    start_nodes, _ = coords_to_nodes(skel=skel, query_coords=starts_coords)
    end_nodes, _ = coords_to_nodes(skel=skel, query_coords=ends_coords)

    bulb_starts_on_path = []
    bulb_ends_on_path = []
    for i in range(len(starts_coords)):
        if (start_nodes[i] in endpoint_path_nodes) and (end_nodes[i] in endpoint_path_nodes):
            bulb_starts_on_path.append(starts_coords[i])
            bulb_ends_on_path.append(ends_coords[i])

    return np.array(bulb_starts_on_path), np.array(bulb_ends_on_path)


