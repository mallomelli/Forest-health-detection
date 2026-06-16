import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import grid_from_df


def compute_green_cover(img, tile_size=64, veg_threshold=20, valid_threshold=30):
    h, w = img.shape[:2]
    records = []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = img[y:min(y+tile_size, h), x:min(x+tile_size, w)]
            R = tile[:, :, 0].astype(float)
            G = tile[:, :, 1].astype(float)
            B = tile[:, :, 2].astype(float)
            exg = 2 * G - R - B
            valid = (R + G + B) > valid_threshold
            if valid.sum() == 0:
                coverage = np.nan
            else:
                coverage = (exg > veg_threshold).sum() / valid.sum() * 100
            records.append({"x": x, "y": y, "coverage_pct": coverage})
    return pd.DataFrame(records)


def save_green_heatmap(df, out_png, threshold=40):
    grid, xs, ys, x_idx, y_idx = grid_from_df(df, "coverage_pct")
    alerts = df[df["coverage_pct"] < threshold]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="Vegetation Coverage (%)")

    alert_rows = [y_idx[r] for r in alerts["y"]]
    alert_cols = [x_idx[c] for c in alerts["x"]]
    ax.scatter(alert_cols, alert_rows, c="red", s=20,
               label=f"Low veg alert (<{threshold}%)", zorder=5)
    ax.legend(fontsize=11)
    ax.set_title(
        "Vegetation Coverage Heatmap\n(green = high coverage, red = low)")
    ax.set_xlabel("Tile Column")
    ax.set_ylabel("Tile Row")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")
