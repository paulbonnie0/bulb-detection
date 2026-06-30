from typing import Any, Tuple, List, Union
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, peak_widths


def evenly_sample_branch(
        geodesic_dists: np.ndarray,
        radii: np.ndarray,
        step_size: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate the radii to be evenly sampled."""

    sort_idx = np.argsort(geodesic_dists)
    geodesic_dists_sorted = geodesic_dists[sort_idx]
    radii_sorted = radii[sort_idx]

    even_dists = np.arange(geodesic_dists_sorted[0], geodesic_dists_sorted[-1] + step_size, step_size)
    even_radii = np.interp(even_dists, geodesic_dists_sorted, radii_sorted)

    return even_dists, even_radii


def compute_mad_threshold(
        signal: np.ndarray,
        k: float = 2.0
) -> float:
    """Computes the dynampic peak threshold using the MAD."""

    signal_median = np.median(signal)
    mad = np.median(np.abs(signal - signal_median))
    sigma = 1.4826 * mad
    dynamic_threshold = signal_median + (k * sigma)

    return dynamic_threshold


def detect_branch_peaks(
        even_dists: np.ndarray,
        even_radii: np.ndarray,
        window_size: float = 15.0,
        k: float = 2.0,
        min_peak_dist_um: float | None = None,
        rel_height: float = 0.5
) -> Tuple[List[Tuple[float, float]], float]:
    """
    Applies median baseline subtraction and find_peaks to a branch.
    Returns the spans of the detected bulbs and the dynamic threshold used.
    """
    step_size = even_dists[1] - even_dists[0]
    window_samples = int(window_size / step_size)
    if min_peak_dist_um is not None:
        min_peak_dist_samples = int(min_peak_dist_um / step_size)
    else:
        min_peak_dist_samples = None

    baseline_radii = median_filter(even_radii, size=window_samples, mode='reflect')
    radii_filtered = even_radii - baseline_radii
    dynamic_threshold = compute_mad_threshold(signal=radii_filtered, k=k)
    peaks, _ = find_peaks(radii_filtered, height=dynamic_threshold, distance=min_peak_dist_samples)

    if len(peaks) == 0:
        return [], dynamic_threshold

    _, _, left_ips, right_ips = peak_widths(radii_filtered, peaks, rel_height=rel_height)

    peak_spans = []
    for left, right in zip(left_ips, right_ips):
        start_dist = even_dists[0] + (left * step_size)
        end_dist = even_dists[0] + (right * step_size)
        peak_spans.append((start_dist, end_dist))

    return peak_spans, dynamic_threshold


def predict_bulbs(
        skel: Any,
        geodesic_dists: np.ndarray,
        radii: np.ndarray,
        **detection_kwargs
) -> Tuple[List[List[int]], List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    """
    1. Find all leaf nodes (degree 1)
    2. Run detect_branch_peaks on every root to leaf node path
    3. Map the distance spans back to the skeleton node_ids
    4. Use induced subgraphs to group the nodes into bulbs
    """

    # 1.
    leaf_nodes = skel.nodes_of_degree(k=1)
    skel_graph = skel._igraph()
    all_paths = skel_graph.get_shortest_paths(0, to=leaf_nodes)

    # 2.
    all_bulb_nodes = set()
    branch_thresholds = []
    for path in all_paths:
        path_nodes = path[1:]
        path_dists = geodesic_dists[path_nodes]
        path_radii = radii[path_nodes]

        even_dists, even_radii = evenly_sample_branch(path_dists, path_radii)

        peak_spans, dynamic_threshold = detect_branch_peaks(even_dists, even_radii, **detection_kwargs)
        branch_thresholds.append(dynamic_threshold)

        # 3.
        for start_dist, end_dist in peak_spans:
            bulb_mask = (path_dists >= start_dist) & (path_dists <= end_dist)
            bulb_node_ids = np.array(path_nodes)[bulb_mask]
            all_bulb_nodes.update(bulb_node_ids)
        
    all_bulb_nodes_list = sorted(list(all_bulb_nodes))

    # 4. 
    bulb_subgraph = skel_graph.induced_subgraph(all_bulb_nodes_list)
    components = bulb_subgraph.connected_components()

    predicted_bulb_clusters = []
    predicted_bulb_coords = []

    for component in components:
        original_cluster_ids = [all_bulb_nodes_list[v] for v in component]
        predicted_bulb_clusters.append(original_cluster_ids)

        cluster_distances = geodesic_dists[original_cluster_ids]
        start_idx = np.argmin(cluster_distances)
        end_idx = np.argmax(cluster_distances)
        start_node_id = original_cluster_ids[start_idx]
        end_node_id = original_cluster_ids[end_idx]

        start_coords = skel.nodes[start_node_id]
        end_coords = skel.nodes[end_node_id]

        predicted_bulb_coords.append((start_coords, end_coords))

    return predicted_bulb_clusters, predicted_bulb_coords, branch_thresholds


def merge_split_clusters(
        skel: Any,
        geodesic_dists: np.ndarray,
        clusters: List[List[int]],
        merge_threshold_um: float = 1.0
) -> List[List[int]]:
    """Merge clusters that are below the merge_threshold_um apart."""

    skel_graph = skel._igraph()

    node_to_cluster = {}
    for cluster_idx, cluster in enumerate(clusters):
        for node in cluster:
            node_to_cluster[node] = cluster_idx
    
    merge_groups = [{i} for i in range(len(clusters))]

    for child_idx, cluster in enumerate(clusters):
        cluster_dists = geodesic_dists[cluster]
        node_closest_to_soma = cluster[np.argmin(cluster_dists)]

        path_to_soma = skel_graph.get_shortest_paths(0, to=node_closest_to_soma)[0]

        for node in reversed(path_to_soma[:-1]):
            if node in node_to_cluster:
                parent_idx = node_to_cluster[node]

                if parent_idx != child_idx:
                    boundary_node = node
                    dist = geodesic_dists[node_closest_to_soma] - geodesic_dists[boundary_node]

                    if dist < merge_threshold_um:
                        group1 = merge_groups[child_idx]
                        group2 = merge_groups[parent_idx]

                        if group1 is not group2:
                            new_group = group1.union(group2)
                            for idx in new_group:
                                merge_groups[idx] = new_group
                    break
            
    unique_groups = set(frozenset(group) for group in merge_groups)

    merged_bulb_clusters = []
    for group in unique_groups:
        merged_nodes = []
        for cluster_idx in group:
            merged_nodes.extend(clusters[cluster_idx])
        
        merged_bulb_clusters.append(merged_nodes)
    
    return merged_bulb_clusters

