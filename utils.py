from pathlib import Path
import numpy as np
from PIL import Image


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_image_rgb(image_path):
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def grid_from_df(df, value_col):
    xs = sorted(df["x"].unique())
    ys = sorted(df["y"].unique())
    x_idx = {v: i for i, v in enumerate(xs)}
    y_idx = {v: i for i, v in enumerate(ys)}
    grid = np.full((len(ys), len(xs)), np.nan)
    for _, row in df.iterrows():
        grid[y_idx[row["y"]], x_idx[row["x"]]] = row[value_col]
    return grid, xs, ys, x_idx, y_idx
