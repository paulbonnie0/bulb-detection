from typing import Any, Tuple, List, Union, Set, Optional
import numpy as np
import skeliner as sk
import matplotlib.pyplot as plt
import random



DEFAULT_SKEL_CMAP = {
"soma": "#808080",
"axon": "#808080",
"dendrite": "#808080",
"apical": "#808080",
"fork": "#808080",
"end": "#808080",
"root": "#808080",
"undefined": "#808080"
}

def plot_anns_on_skeleton(
        skel: Any,
        ann_stars: np.ndarray,
        ann_ends: np.ndarray,
        ax: Optional[plt.Axes] = None,
        ann_color: str = "#00BFFF",
        label: str = 'annotated bulbs',
        title: str = '',
        skel_cmap: Optional[dict[str, str]] = None
) -> plt.Axes:
    """Plot the 2d skeleton with bulb annotations."""
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    if skel_cmap is None:
        skel_cmap = DEFAULT_SKEL_CMAP

    sk.plot2d(skel, ax=ax, unit='μm', color_by='ntype', skel_cmap=skel_cmap)

    for i in range(ann_stars.shape[0]):
        x = [ann_stars[i, 0], ann_ends[i, 0]]
        y = np.array([ann_stars[i, 1], ann_ends[i, 1]])
        if i == 0:
            ax.plot(x, y, ann_color, lw=2.5, label=label)
        else:
            ax.plot(x, y, ann_color, lw=2.5)
    ax.legend(frameon=False)
    ax.set_title(title)
    
    return ax


def plot_branch_radius_profile(
        geodesic_dists: np.ndarray,
        radii: np.ndarray,
        start_ids: np.ndarray,
        end_ids: np.ndarray,
        ax: Optional[plt.Axes],
        title: str = '',
        xmin: Optional[float] = None,
        xmax: Optional[float] = None 
) -> plt.Axes:
    """Plot the radii vs geodesic distance with marked bulbs for a single branch."""
     
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 6))
    
    sort_idx = np.argsort(geodesic_dists)
    geodesic_dists_sorted = geodesic_dists[sort_idx][1:]
    radii_sorted = radii[sort_idx][1:]

    ax.plot(geodesic_dists_sorted, radii_sorted)

    cmap = plt.get_cmap('turbo')
    cluster_colors = [cmap(i) for i in np.linspace(0, 1, len(start_ids))]
    random.shuffle(cluster_colors)

    for i in range(len(start_ids)):
        cluster_start = geodesic_dists[start_ids[i]]
        cluster_end = geodesic_dists[end_ids[i]]
        plt.axvspan(cluster_start, cluster_end, color=cluster_colors[i], alpha=0.3)
    
    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin, xmax)
    elif xmin is not None:
        ax.set_xlim(xmin, np.max(geodesic_dists_sorted))
    elif xmax is not None:
        ax.set_xlim(np.min(geodesic_dists_sorted), xmax)
    else:
        ax.set_xlim(np.min(geodesic_dists_sorted), np.max(geodesic_dists_sorted))

    ax.set_xlabel("Geodesic Distance to Soma (μm)")
    ax.set_ylabel("Radius (μm)")
    ax.set_title(title)

    return ax


def plot_evaluation_on_skeleton(
        skel: Any,
        tp_nodes: np.ndarray,
        fp_nodes: np.ndarray,
        fn_nodes: np.ndarray,
        ax : Optional[plt.Axes] = None,
        tp_color: str = '#2CBA00',
        fp_color: str = '#FFA500',
        fn_color: str = '#CC2936',
        s: Optional[int] = 20,
        skel_cmap: Optional[dict[str, str]] = None
) -> plt.Axes:
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    if skel_cmap is None:
        skel_cmap = DEFAULT_SKEL_CMAP
    
    sk.plot2d(skel, ax=ax, unit='μm', color_by='ntype', skel_cmap=skel_cmap)

    ax.scatter(skel.nodes[tp_nodes, 0], skel.nodes[tp_nodes, 1], color=tp_color, s=s, label="True positives", zorder=7)
    ax.scatter(skel.nodes[fp_nodes, 0], skel.nodes[fp_nodes, 1], color=fp_color, s=s, label="False positives", zorder=6)
    ax.scatter(skel.nodes[fn_nodes, 0], skel.nodes[fn_nodes, 1], color=fn_color, s=s, label="False negatives", zorder=5)

    ax.legend(frameon=False)
    ax.set_title("Bulb Classification")

    return ax

    