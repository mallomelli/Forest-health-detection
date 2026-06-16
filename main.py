
from pathlib import Path
from config import (
    IMAGE_PATH,
    OUTPUT_DIR,
    TILE_SIZE,
    COVER_THRESHOLD,
    TREE_THRESHOLD,
    BURN_THRESHOLD,
    SCORE_THRESHOLD,
    USE_RECTIFY,
    RECTIFY_MODE,
    MANUAL_ANGLE,
    RECTIFIED_IMAGE_PATH,
)
from utils import ensure_dir, load_image_rgb
from vegetation import compute_green_cover, save_green_heatmap
from burn_detection import compute_burn_score, save_burn_heatmap
from tree_detection import run_deepforest, annotate_image, compute_tree_density, save_density_heatmap
from combined_map import save_combined_map, save_burn_map
from rectify import auto_deskew, manual_deskew
from config import VIDEO_PATH, VIDEO_OUTPUT_DIR, FRAME_INTERVAL_SEC
from video_pipeline import process_video


def main():
    ensure_dir(OUTPUT_DIR)
    out = Path(OUTPUT_DIR)

    image_path_to_use = IMAGE_PATH

    if USE_RECTIFY:
        print("Rectifying image...")

        if RECTIFY_MODE.lower() == "auto":
            image_path_to_use = auto_deskew(
                IMAGE_PATH,
                out_path=RECTIFIED_IMAGE_PATH
            )
        elif RECTIFY_MODE.lower() == "manual":
            image_path_to_use = manual_deskew(
                IMAGE_PATH,
                angle_degrees=MANUAL_ANGLE,
                out_path=RECTIFIED_IMAGE_PATH
            )
        else:
            raise ValueError("RECTIFY_MODE must be 'auto' or 'manual'")

        print(f"Using rectified image: {image_path_to_use}")

    print("Loading image...")
    img = load_image_rgb(image_path_to_use)

    print("Computing vegetation coverage...")
    green_df = compute_green_cover(img, tile_size=TILE_SIZE)
    green_df.to_csv(out / "vegetation_coverage.csv", index=False)
    save_green_heatmap(
        green_df,
        out / "vegetation_coverage_heatmap.png",
        threshold=COVER_THRESHOLD
    )

    print("Computing burn area...")
    burn_df = compute_burn_score(img, tile_size=TILE_SIZE)
    burn_df.to_csv(out / "burn_scores.csv", index=False)

    save_burn_heatmap(
        burn_df,
        out / "burn_heatmap.png",
        threshold=BURN_THRESHOLD
    )

    save_burn_map(
        burn_df,
        out / "burn_condition_map.png",
        burn_threshold=BURN_THRESHOLD
    )

    print("Running DeepForest on CPU...")
    predictions = run_deepforest(
        image_path_to_use,
        score_threshold=SCORE_THRESHOLD
    )
    predictions.to_csv(out / "deepforest_predictions.csv", index=False)

    annotate_image(
        image_path_to_use,
        predictions,
        out / "deepforest_annotated.png",
        score_threshold=SCORE_THRESHOLD
    )

    print("Computing tree density...")
    density_df = compute_tree_density(img, predictions, tile_size=TILE_SIZE)
    density_df.to_csv(out / "tree_density.csv", index=False)

    save_density_heatmap(
        density_df,
        out / "tree_density_heatmap.png",
        threshold=TREE_THRESHOLD
    )

    print("Building combined forest condition map...")
    save_combined_map(
        green_df,
        density_df,
        out / "combined_forest_condition.png",
        canopy_threshold=COVER_THRESHOLD,
        tree_threshold=TREE_THRESHOLD
    )

    print(f"\nAll done. Outputs saved to: {OUTPUT_DIR}/")
    print("\nOutputs summary:")
    print("  vegetation_coverage_heatmap.png  — canopy coverage")
    print("  burn_heatmap.png                 — burn intensity")
    print("  burn_condition_map.png           — burn area map (standalone)")
    print("  deepforest_annotated.png         — detected trees")
    print("  tree_density_heatmap.png         — tree density per tile")
    print("  combined_forest_condition.png    — forest health (canopy + trees only)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--video":
        # Run video mode
        process_video(
            video_path=VIDEO_PATH,
            output_dir=VIDEO_OUTPUT_DIR,
            frame_interval_sec=FRAME_INTERVAL_SEC,
            tile_size=TILE_SIZE,
            cover_threshold=COVER_THRESHOLD,
            tree_threshold=TREE_THRESHOLD,
            burn_threshold=BURN_THRESHOLD,
            score_threshold=SCORE_THRESHOLD,
        )
    else:
        # Run single image mode as before
        main()
