
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from utils import grid_from_df


def run_deepforest(image_path, score_threshold=0.4):
    from deepforest import main as deepforest_main

    model = deepforest_main.deepforest()
    model.load_model(model_name="weecology/deepforest-tree", revision="main")
    model.to("cpu")
    print("DeepForest model loaded on CPU")

    predictions = model.predict_tile(
        path=str(image_path),
        patch_size=400,
        patch_overlap=0.05,
        iou_threshold=0.15,
    )

    if predictions is None or len(predictions) == 0:
        return pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "score", "label"])

    predictions = predictions[predictions["score"] >= score_threshold].copy()
    return predictions


def annotate_image(image_path, predictions, out_path, score_threshold=0.4):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    if predictions is not None and len(predictions) > 0:
        filtered = predictions[predictions["score"] >= score_threshold]
        for _, row in filtered.iterrows():
            x1 = float(row["xmin"])
            y1 = float(row["ymin"])
            x2 = float(row["xmax"])
            y2 = float(row["ymax"])
            score = float(row["score"])

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, max(0, y1 - 12)), f"{score:.2f}", fill="red")

    img.save(out_path)
    print(f"Saved: {out_path}")


def compute_tree_density(img, predictions, tile_size=64):
    h, w = img.shape[:2]
    records = []

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)

            count = 0
            for _, row in predictions.iterrows():
                xc = (row["xmin"] + row["xmax"]) / 2
                yc = (row["ymin"] + row["ymax"]) / 2

                if x <= xc < x2 and y <= yc < y2:
                    count += 1

            records.append({
                "x": x,
                "y": y,
                "tree_count": count
            })

    return pd.DataFrame(records)


def save_density_heatmap(df, out_png, threshold=2):
    grid, xs, ys, x_idx, y_idx = grid_from_df(df, "tree_count")
    alerts = df[df["tree_count"] <= threshold]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(grid, cmap="YlGn", vmin=0, vmax=max(1, np.nanmax(grid)))
    plt.colorbar(im, ax=ax, label="Detected Trees per Tile")

    if not alerts.empty:
        alert_rows = [y_idx[r] for r in alerts["y"]]
        alert_cols = [x_idx[c] for c in alerts["x"]]
        ax.scatter(
            alert_cols,
            alert_rows,
            c="red",
            s=20,
            label=f"Low density (≤{threshold})",
            zorder=5
        )
        ax.legend(fontsize=11)

    ax.set_title("Tree Density Heatmap")
    ax.set_xlabel("Tile Column")
    ax.set_ylabel("Tile Row")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")
