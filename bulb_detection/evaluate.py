from typing import Any, Tuple, List, Union, Set
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_IoU(s1: Set[int], s2: Set[int]) -> float:
    """Computes the Intersection over Union for two sets."""
    return len(s1.intersection(s2)) / len(s1.union(s2)) if len(s1.union(s2)) > 0 else 0


def match_clusters(
        gt_clusters: List[List[int]],
        pred_clusters: List[List[int]],
        iou_threshold: float = 0.5
) -> Tuple[List[int], List[int], List[int]]:
    """Compute IoU matrix and find assignments with Hungarian Algorithm."""
    gt_cluster_sets = [set(cluster) for cluster in gt_clusters]
    pred_clusters_sets = [set(cluster) for cluster in pred_clusters]

    iou_matrix = np.zeros((len(gt_cluster_sets), len(pred_clusters_sets)))
    for i, gt_bulb in enumerate(gt_cluster_sets):
        for j, pred_bulb in enumerate(pred_clusters_sets):
            iou_matrix[i, j] = compute_IoU(gt_bulb, pred_bulb)

    gt_ids, pred_ids = linear_sum_assignment(iou_matrix, maximize=True)

    # True positives 
    tp_clusters = []
    gt_match_ids = set() 
    pred_match_ids = set()

    for gt_id, pred_id in zip(gt_ids, pred_ids):
        if iou_matrix[gt_id, pred_id] >= iou_threshold:
            
            tp_clusters.append(pred_clusters[pred_id])
            gt_match_ids.add(gt_id) # Holds the indices of the gt_bulb_sets that are tp 
            pred_match_ids.add(pred_id) # Same but for the pred_bulb_sets

    # False positives (if the set is pred but not gt)
    fp_clusters = []
    for cluster_id, pred_cluster in enumerate(pred_clusters):
        if cluster_id not in pred_match_ids:
            fp_clusters.append(pred_cluster)

    # False negatives (if the set is gt but not pred)
    fn_clusters = []
    for cluster_id, gt_cluster in enumerate(gt_clusters):
        if cluster_id not in gt_match_ids:
            fn_clusters.append(gt_cluster)

    return tp_clusters, fp_clusters, fn_clusters

def get_f1_score(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Returns precision, recall, and f1 score"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'f1_score': f1_score}


def evaluate_detection(
        gt_clusters: List[List[int]],
        pred_clusters: List[List[int]],
        iou_threshold: float = 0.5
) -> dict[str, Any]:
    """"""
    tp_clusters, fp_clusters, fn_clusters = match_clusters(gt_clusters, pred_clusters, iou_threshold)
    tp = len(tp_clusters)
    fp = len(fp_clusters)
    fn = len(fn_clusters)
    metrics = get_f1_score(tp, fp, fn)

    return_dict = {
        **metrics,
        'tp_clusters': tp_clusters,
        'fp_clusters': fp_clusters,
        'fn_clusters': fn_clusters
    }

    return return_dict


