# video_pipeline.py

import cv2
from pathlib import Path
from utils import ensure_dir, load_image_rgb
from vegetation import compute_green_cover, save_green_heatmap
from burn_detection import compute_burn_score, save_burn_heatmap
from tree_detection import run_deepforest, annotate_image, compute_tree_density, save_density_heatmap
from combined_map import save_combined_map, save_burn_map


def process_video(
    video_path,
    output_dir="output_video",
    frame_interval_sec=5,
    tile_size=64,
    cover_threshold=40,
    tree_threshold=2,
    burn_threshold=20,
    score_threshold=0.4,
):
    """
    Extracts frames from a video at a set interval and runs
    the full forest analysis pipeline on each frame.

    Args:
        video_path         : path to your video file (.mp4, .avi, .mov, etc.)
        output_dir         : root folder for all frame outputs
        frame_interval_sec : how many seconds between each analysed frame
        tile_size          : pixel size of each analysis tile
        cover_threshold    : vegetation coverage alert threshold
        tree_threshold     : tree density alert threshold
        burn_threshold     : burn score alert threshold
        score_threshold    : DeepForest confidence cutoff
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    print(f"Video loaded: {video_path}")
    print(f"FPS: {fps:.2f} | Total frames: {total_frames} | Duration: {duration_sec:.1f}s")
    print(f"Analysing one frame every {frame_interval_sec}s")

    frame_step = int(fps * frame_interval_sec)
    frame_indices = list(range(0, total_frames, frame_step))
    print(f"Total frames to analyse: {len(frame_indices)}")

    ensure_dir(output_dir)
    frames_dir = Path(output_dir) / "frames"
    ensure_dir(frames_dir)

    for i, frame_idx in enumerate(frame_indices):
        timestamp_sec = frame_idx / fps
        print(f"\n--- Frame {i+1}/{len(frame_indices)} | t={timestamp_sec:.1f}s ---")

        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"  Could not read frame {frame_idx}, skipping.")
            continue

        # Save frame as temp PNG
        frame_png = frames_dir / f"frame_{i+1:04d}_t{int(timestamp_sec):05d}s.png"
        cv2.imwrite(str(frame_png), frame)

        # Create per-frame output folder
        frame_out = Path(output_dir) / f"frame_{i+1:04d}_t{int(timestamp_sec):05d}s"
        ensure_dir(frame_out)

        # Load as RGB numpy array
        img = load_image_rgb(str(frame_png))

        # --- Vegetation coverage ---
        green_df = compute_green_cover(img, tile_size=tile_size)
        green_df.to_csv(frame_out / "vegetation_coverage.csv", index=False)
        save_green_heatmap(
            green_df,
            frame_out / "vegetation_coverage_heatmap.png",
            threshold=cover_threshold
        )

        # --- Burn detection ---
        burn_df = compute_burn_score(img, tile_size=tile_size)
        burn_df.to_csv(frame_out / "burn_scores.csv", index=False)
        save_burn_heatmap(
            burn_df,
            frame_out / "burn_heatmap.png",
            threshold=burn_threshold
        )
        save_burn_map(
            burn_df,
            frame_out / "burn_condition_map.png",
            burn_threshold=burn_threshold
        )

        # --- DeepForest tree detection ---
        predictions = run_deepforest(
            str(frame_png),
            score_threshold=score_threshold
        )
        predictions.to_csv(frame_out / "deepforest_predictions.csv", index=False)
        annotate_image(
            str(frame_png),
            predictions,
            frame_out / "deepforest_annotated.png",
            score_threshold=score_threshold
        )

        # --- Tree density ---
        density_df = compute_tree_density(img, predictions, tile_size=tile_size)
        density_df.to_csv(frame_out / "tree_density.csv", index=False)
        save_density_heatmap(
            density_df,
            frame_out / "tree_density_heatmap.png",
            threshold=tree_threshold
        )

        # --- Combined condition map ---
        save_combined_map(
            green_df,
            density_df,
            frame_out / "combined_forest_condition.png",
            canopy_threshold=cover_threshold,
            tree_threshold=tree_threshold
        )

        print(f"  Saved outputs to: {frame_out}")

    cap.release()
    print(f"\nAll frames processed. Results in: {output_dir}/")
