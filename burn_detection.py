import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import grid_from_df


def compute_burn_score(img, tile_size=64):
    """
    Detects burned areas using char index:
    burned pixels tend to be dark (low R, G, B) with relatively higher R than G/B.
    Returns a DataFrame with burn_score (0-100%) per tile.
    """
    h, w = img.shape[:2]
    records = []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = img[y:min(y+tile_size, h), x:min(x+tile_size, w)]
            R = tile[:, :, 0].astype(float)
            G = tile[:, :, 1].astype(float)
            B = tile[:, :, 2].astype(float)

            brightness = (R + G + B) / 3.0

            dark = brightness < 60
            char = (R > G) & (R > B)
            burned = dark & char

            total_pixels = tile.shape[0] * tile.shape[1]
            burn_score = (burned.sum() / total_pixels) * \
                100 if total_pixels > 0 else 0.0

            records.append({"x": x, "y": y, "burn_score": burn_score})
    return pd.DataFrame(records)


def save_burn_heatmap(df, out_png, threshold=20):
    grid, xs, ys, x_idx, y_idx = grid_from_df(df, "burn_score")
    alerts = df[df["burn_score"] >= threshold]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(grid, cmap="hot_r", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="Burn Score (%)")

    alert_rows = [y_idx[r] for r in alerts["y"]]
    alert_cols = [x_idx[c] for c in alerts["x"]]
    ax.scatter(alert_cols, alert_rows, c="blue", s=20,
               label=f"Burn alert (≥{threshold}%)", zorder=5)
    ax.legend(fontsize=11)
    ax.set_title("Burn Area Heatmap\n(dark=low burn, bright=high burn)")
    ax.set_xlabel("Tile Column")
    ax.set_ylabel("Tile Row")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")
