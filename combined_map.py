import numpy as np
import matplotlib.pyplot as plt
from utils import grid_from_df


def save_combined_map(green_df, density_df, out_png,
                      canopy_threshold=40, tree_threshold=2):
    canopy_grid, *_ = grid_from_df(green_df, "coverage_pct")
    density_grid, xs, ys, x_idx, y_idx = grid_from_df(density_df, "tree_count")

    risk_grid = np.full_like(canopy_grid, np.nan)
    for i in range(canopy_grid.shape[0]):
        for j in range(canopy_grid.shape[1]):
            c = canopy_grid[i, j]
            t = density_grid[i, j]
            if np.isnan(c):
                risk_grid[i, j] = np.nan
            elif c < canopy_threshold and t <= tree_threshold:
                risk_grid[i, j] = 2
            elif c < canopy_threshold or t <= tree_threshold:
                risk_grid[i, j] = 1
            else:
                risk_grid[i, j] = 0

    risk_colors = np.zeros((*risk_grid.shape, 3), dtype=np.uint8)
    risk_colors[risk_grid == 0] = [0, 180, 0]
    risk_colors[risk_grid == 1] = [255, 200, 0]
    risk_colors[risk_grid == 2] = [220, 0, 0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im1 = axes[0].imshow(canopy_grid, cmap="RdYlGn", vmin=0, vmax=100)
    axes[0].set_title("Vegetation Coverage (%)\n(Green=high, Red=low)")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(density_grid, cmap="YlGn",
                         vmin=0, vmax=max(1, np.nanmax(density_grid)))
    axes[1].set_title("Tree Density\n(DeepForest detections)")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(risk_colors)
    axes[2].set_title(
        "Forest Condition\n(Canopy + Tree Density only)\nGreen=OK  Yellow=Warning  Red=Critical")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Forest Health — Combined Condition Map",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")


def save_burn_map(burn_df, out_png, burn_threshold=20):
    grid, xs, ys, x_idx, y_idx = grid_from_df(burn_df, "burn_score")
    alerts = burn_df[burn_df["burn_score"] >= burn_threshold]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(grid, cmap="hot_r", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="Burn Score (%)")

    alert_rows = [y_idx[r] for r in alerts["y"]]
    alert_cols = [x_idx[c] for c in alerts["x"]]
    ax.scatter(alert_cols, alert_rows, c="blue", s=20,
               label=f"Burn alert (≥{burn_threshold}%)", zorder=5)
    ax.legend(fontsize=11)
    ax.set_title("Burn Area Detection Map\n(Separate from Forest Condition)")
    ax.set_xlabel("Tile Column")
    ax.set_ylabel("Tile Row")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")
