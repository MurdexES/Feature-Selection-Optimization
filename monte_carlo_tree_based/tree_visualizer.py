"""Matplotlib snapshots of an MCTS feature-selection tree."""

from __future__ import annotations

import textwrap
from collections import defaultdict
from heapq import heappop, heappush
from itertools import count
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .MCTS import MCTSNode
except ImportError:
    from MCTS import MCTSNode


def _short(label: str, width: int = 22) -> str:
    return textwrap.shorten(str(label), width=width, placeholder="…")


def _visible_nodes(root: MCTSNode, max_nodes: int) -> tuple[list[MCTSNode], dict]:
    """Return an ancestor-complete view prioritising the most visited branches."""
    if max_nodes < 1:
        raise ValueError("max_nodes must be at least 1")

    nodes = [root]
    depth = {root: 0}
    order = count()
    frontier = []

    def add_children(node):
        for child in node.children:
            # The counter makes ties deterministic and avoids comparing node objects.
            heappush(frontier, (-child.visits, next(order), child))

    add_children(root)
    while frontier and len(nodes) < max_nodes:
        _, _, node = heappop(frontier)
        nodes.append(node)
        depth[node] = depth[node.parent] + 1
        add_children(node)

    return nodes, depth


def _node_label(node, root, reward_to_mae, dense):
    """Build either a detailed label or a compact label for crowded trees."""
    mae = reward_to_mae(node.mean_reward) if node.visits else float("nan")
    if node is root:
        return f"ROOT\nvisits={node.visits}  est.MAE={mae:.4f}"

    feature = _short(node.game_state.feature_label(node.move), 20 if dense else 24)
    if dense:
        return f"+{feature}\nv={node.visits}  MAE={mae:.4f}"

    labels = [_short(label, 15) for label in node.game_state.selected_labels()]
    subset = "subset: [" + ", ".join(labels) + "]"
    return f"+ {feature}\n{subset}\nvisits={node.visits}  est.MAE={mae:.4f}"


def save_tree_snapshot(
    root: MCTSNode,
    iteration: int,
    output_dir: Path,
    reward_to_mae,
    best_subset=(),
    max_nodes: int = 180,
) -> Path:
    """Save a top-down, landscape PNG showing the tree accumulated so far."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes, depth = _visible_nodes(root, max_nodes)
    included = set(nodes)
    children = defaultdict(list)
    for node in nodes:
        if node.parent in included:
            children[node.parent].append(node)

    x_pos = {}
    leaf_counter = [0]

    def place(node):
        visible_children = sorted(
            children.get(node, []),
            key=lambda child: (child.mean_reward, child.visits),
            reverse=True,
        )
        if not visible_children:
            x_pos[node] = leaf_counter[0]
            leaf_counter[0] += 1
            return
        for child in visible_children:
            place(child)
        x_pos[node] = float(np.mean([x_pos[c] for c in visible_children]))

    place(root)
    max_depth = max(depth.values(), default=0)
    leaf_count = max(leaf_counter[0], 1)
    # Detailed three-line labels work for genuinely small trees. Once a level
    # becomes broad, compact rotated labels preserve one readable label per node.
    dense = len(nodes) > 35 or leaf_count > 14

    # Width follows the number of visible leaves, while height follows tree depth.
    # The lower bound on width guarantees a landscape image even for a young tree.
    fig_h = min(20.0, max(8.0, 4.5 + max_depth * (2.0 if dense else 2.5)))
    fig_w = min(42.0, max(14.0, leaf_count * (0.36 if dense else 0.85)))
    fig_w = max(fig_w, min(42.0, fig_h * 1.45))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for parent, child_nodes in children.items():
        for child in child_nodes:
            ax.plot(
                [x_pos[parent], x_pos[child]],
                [-depth[parent], -depth[child]],
                color="#b0bec5",
                lw=max(0.5, min(2.5, child.visits / max(root.visits, 1) * 8)),
                zorder=1,
            )

    values = np.array([node.mean_reward for node in nodes], dtype=float)
    sizes = [55 + 260 * np.sqrt(node.visits / max(root.visits, 1)) for node in nodes]
    scatter = ax.scatter(
        [x_pos[n] for n in nodes],
        [-depth[n] for n in nodes],
        c=values,
        cmap="viridis",
        s=sizes,
        edgecolor="#263238",
        linewidth=0.5,
        zorder=3,
    )

    best_key = tuple(sorted(best_subset))
    for node in nodes:
        label = _node_label(node, root, reward_to_mae, dense)
        color = "#b71c1c" if node.game_state.get_state() == best_key else "#263238"
        rotation = 58 if dense and node is not root else 0
        ax.annotate(
            label,
            (x_pos[node], -depth[node]),
            xytext=(0, -7 if node is not root else 8),
            textcoords="offset points",
            fontsize=5.0 if dense else 6.2,
            color=color,
            rotation=rotation,
            rotation_mode="anchor",
            ha="right" if rotation else "center",
            va="top" if node is not root else "bottom",
            bbox=(
                dict(facecolor="white", edgecolor="none", alpha=0.68, pad=0.2)
                if dense else None
            ),
            zorder=4,
        )

    fig.colorbar(
        scatter,
        ax=ax,
        pad=0.015,
        shrink=0.82,
        label="mean rollout reward (higher is better)",
    )
    omitted = max(0, _count_nodes(root) - len(nodes))
    suffix = f"; {omitted} low-visit nodes omitted" if omitted else ""
    ax.set_title(
        f"MCTS feature-selection tree after {iteration} iterations{suffix}\n"
        "Root is at the top; each edge adds the feature named at its child"
        + (" (compact labels used for readability)" if dense else ""),
        fontsize=11,
    )
    ax.set_xticks([])
    ax.set_yticks([-level for level in range(max_depth + 1)])
    ax.set_yticklabels([str(level) for level in range(max_depth + 1)])
    ax.set_ylabel("Tree depth / explicitly selected features")
    ax.margins(x=0.025)
    ax.set_ylim(-max_depth - (0.72 if dense else 0.55), 0.45)
    if best_subset:
        labels = [root.game_state.feature_label(i) for i in best_subset]
        best_text = "Best complete rollout subset:\n" + "\n".join(
            textwrap.wrap(", ".join(labels), width=75)
        )
        ax.text(
            0.005,
            0.995,
            best_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7,
            color="#b71c1c",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9,
                      edgecolor="#ef9a9a"),
            zorder=8,
        )
    path = output_dir / f"tree_step_{iteration:05d}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _count_nodes(root: MCTSNode) -> int:
    count = 0
    stack = [root]
    while stack:
        node = stack.pop()
        count += 1
        stack.extend(node.children)
    return count
