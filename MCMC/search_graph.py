import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def _short(names, i):
    s = str(names[i]) if names is not None else str(i)
    return s.replace("feature_", "f")

# Label show which feature was replaced and which took it place
def _swap_label(frm, prop, names): 
    dropped = frm - prop
    added = prop - frm
    d = _short(names, next(iter(dropped))) if dropped else "?"
    a = _short(names, next(iter(added))) if added else "?"
    return f"-{d} +{a}"


def save_search_graph(edges, node_mae, step, out_dir, names=None, window=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    e = edges[-window:] if window else edges
    if not e:
        return None

    children = defaultdict(list)
    mae_of, is_reject, label_of, depth = {}, {}, {}, {}
    seen = set()

    root = e[0][0]
    seen.add(root); depth[root] = 0
    mae_of[root] = node_mae.get(root, np.nan); is_reject[root] = False; label_of[root] = "start"

    for k, (frm, prop, acc) in enumerate(e):
        if frm not in seen:            # windowed: parent scrolled off -> reattach to root
            seen.add(frm); depth[frm] = 1; children[root].append(frm)
            mae_of[frm] = node_mae.get(frm, np.nan); is_reject[frm] = False; label_of[frm] = "…"
        if acc:
            if prop not in seen:       # newly discovered state -> a real branch
                seen.add(prop); depth[prop] = depth[frm] + 1
                children[frm].append(prop)
                mae_of[prop] = node_mae.get(prop, np.nan); is_reject[prop] = False
                label_of[prop] = _swap_label(frm, prop, names)
            # else: revisit (cycle) -> not representable in a tree, skip
        else:                          # rejected candidate -> unique dead-end leaf
            leaf = ("rej", k)
            depth[leaf] = depth[frm] + 1
            children[frm].append(leaf)
            mae_of[leaf] = node_mae.get(prop, np.nan); is_reject[leaf] = True
            label_of[leaf] = _swap_label(frm, prop, names)

    
    ypos = {}
    counter = [0]
    sys.setrecursionlimit(max(3000, len(e) * 4))

    def place(n):
        ch = children.get(n)
        if not ch:
            ypos[n] = counter[0]; counter[0] += 1
        else:
            for c in ch:
                place(c)
            ypos[n] = float(np.mean([ypos[c] for c in ch]))
    place(root)

    # it is drawing part
    vals = [v for v in mae_of.values() if np.isfinite(v)]
    vmin, vmax = (np.percentile(vals, 5), np.percentile(vals, 95)) if vals else (0.0, 1.0)

    n_leaves = max(counter[0], 1)
    max_depth = max(depth.values())
    fig_w = min(30.0, 7.0 + max_depth * 0.32)
    fig_h = min(24.0, 3.5 + n_leaves * 0.22)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for n, ch in children.items():
        for c in ch:
            ax.plot([depth[n], depth[c]], [ypos[n], ypos[c]], color="0.8", lw=0.5, zorder=1)

    acc_nodes = [n for n in mae_of if not is_reject[n]]
    rej_nodes = [n for n in mae_of if is_reject[n]]

    sc = ax.scatter([depth[n] for n in acc_nodes], [ypos[n] for n in acc_nodes],
                    c=[mae_of[n] for n in acc_nodes], cmap="viridis_r", vmin=vmin, vmax=vmax,
                    s=70, edgecolor="0.3", linewidths=0.4, zorder=3, label="visited (accepted)")
    if rej_nodes:
        ax.scatter([depth[n] for n in rej_nodes], [ypos[n] for n in rej_nodes],
                   c=[mae_of[n] for n in rej_nodes], cmap="viridis_r", vmin=vmin, vmax=vmax,
                   s=28, marker="x", linewidths=1.0, zorder=2, label="rejected (dead-end)")

    # labels: the -dropped +added swap tried at each node
    for n in mae_of:
        if n == root:
            ax.annotate("start", (depth[n], ypos[n]), textcoords="offset points",
                        xytext=(-5, 8), fontsize=6, ha="right", color="0.25")
        elif is_reject[n]:
            ax.annotate(label_of[n], (depth[n], ypos[n]), textcoords="offset points",
                        xytext=(6, 0), fontsize=5.5, ha="left", va="center", color="#b71c1c")
        else:
            ax.annotate(label_of[n], (depth[n], ypos[n]), textcoords="offset points",
                        xytext=(0, 7), fontsize=5.5, ha="center", color="0.2")

    ax.scatter([depth[root]], [ypos[root]], marker=">", s=150, color="#37474f",
               zorder=4, label="root (start)")
    best = min(acc_nodes, key=lambda n: mae_of[n] if np.isfinite(mae_of[n]) else np.inf)
    ax.scatter([depth[best]], [ypos[best]], marker="o", s=150, color="#e53935",
               edgecolor="darkred", linewidths=1.0, zorder=6,
               label=f"best (MAE {mae_of[best]:.4f})")

    fig.colorbar(sc, ax=ax, label="value error MAE (brighter = lower = better)")
    ax.set_xlabel("tree depth  =  # accepted moves from the start")
    ax.set_yticks([])
    ax.set_title(f"MCMC search tree up to step {step}   "
                 f"(node labels = the  -dropped +added  swap tried there)", fontsize=11)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    path = out_dir / f"search_step_{step:04d}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
