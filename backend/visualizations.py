import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torch
from PIL import Image, ImageOps
import torchvision.transforms as tfm
import time
from pathlib import Path
from light_glue.lightglue.lightglue import LightGlue
from light_glue.lightglue.superpoint import SuperPoint
from light_glue.run import light_glue_checker

LIGHTGLUE_AVAILABLE = True

# Height and width of a single image for visualization
IMG_HW = 512
TEXT_H = 175
FONTSIZE = 50
SPACE = 50  # Space between two images


def write_labels_to_image(labels=["text1", "text2"]):
    """Creates an image with text, supporting multi-line labels"""
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FONTSIZE)
    # Use smaller font for distance text
    small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FONTSIZE - 10)
    img = Image.new("RGB", ((IMG_HW * len(labels)) + 50 * (len(labels) - 1), TEXT_H), (1, 1, 1))
    d = ImageDraw.Draw(img)
    for i, text in enumerate(labels):
        # Handle multi-line text (e.g., "Pred0\ndist: 0.123")
        lines = text.split('\n')
        x_pos = (IMG_HW + SPACE) * i + IMG_HW // 2
        
        # Calculate total height of all lines
        line_heights = []
        for line in lines:
            _, _, _, h = d.textbbox((0, 0), line, font=font if line != lines[-1] or 'dist:' not in line else small_font)
            line_heights.append(h)
        
        total_height = sum(line_heights) + (len(lines) - 1) * 5  # 5px spacing between lines
        y_start = (TEXT_H - total_height) // 2
        
        # Draw each line
        y_offset = y_start
        for line_idx, line in enumerate(lines):
            current_font = small_font if 'dist:' in line else font
            _, _, w, h = d.textbbox((0, 0), line, font=current_font)
            d.text((x_pos - w // 2, y_offset), line, fill=(0, 0, 0), font=current_font)
            y_offset += h + 5  # Move down for next line
    
    return Image.fromarray(np.array(img)[:TEXT_H] * 255)


def draw_box(img, c=(0, 1, 0), thickness=20):
    """Draw a colored box around an image. Image should be a PIL.Image."""
    assert isinstance(img, Image.Image)
    img = tfm.ToTensor()(img)
    assert len(img.shape) >= 2, f"{img.shape=}"
    c = torch.tensor(c).type(torch.float).reshape(3, 1, 1)
    img[..., :thickness, :] = c
    img[..., -thickness:, :] = c
    img[..., :, -thickness:] = c
    img[..., :, :thickness] = c
    return tfm.ToPILImage()(img)


def build_prediction_image(images_paths, label, preds_correct, distances=None, num_matches=None):
    """Build images in a grid layout: query first, then predictions.
    If there are more than 4 predictions, arrange them in multiple rows (4 per row).
    
    Parameters
    ----------
    images_paths : list of image paths
    label : bool or None, if None, no labels will be shown
    preds_correct : list of bool/None indicating if prediction is correct (green) or not (red)
    distances : list of float, optional, distances for each prediction to display
    num_matches : list of int, optional, number of LightGlue matches for each prediction
    """
    assert len(images_paths) == len(preds_correct)
    num_predictions = len(preds_correct) - 1  # Exclude query
    
    # Only create labels if label is not None
    labels = []
    if label is not None:
        labels = ["Query"]
        for i, is_correct in enumerate(preds_correct[1:]):
            # Build label components
            label_parts = []
            
            # Prediction number and status
            if is_correct is None:
                label_parts.append(f"Pred{i}")
            elif is_correct:
                label_parts.append(f"Pred{i} ✓")
            else:
                label_parts.append(f"Pred{i} ✗")
            
            # Add match count if available
            if num_matches is not None and i < len(num_matches) and num_matches[i] is not None:
                label_parts.append(f"matches: {num_matches[i]}")
            
            # Add distance if available
            if distances is not None and i < len(distances):
                label_parts.append(f"dist: {distances[i]:.4f}")
            
            # Join all parts with newlines
            labels.append("\n".join(label_parts))

    images = [Image.open(path).convert("RGB") for path in images_paths]
    for img_idx, (img, is_correct) in enumerate(zip(images, preds_correct)):
        if is_correct is None:
            continue
        color = (0, 1, 0) if is_correct else (1, 0, 0)
        img = draw_box(img, color)
        images[img_idx] = img

    resized_images = [tfm.Resize(510, max_size=IMG_HW, antialias=True)(img) for img in images]
    resized_images = [ImageOps.pad(img, (IMG_HW, IMG_HW), color='white') for img in resized_images]  # Apply padding to make them squared

    # Arrange images in grid layout
    if num_predictions > 4:
        # Grid layout: query in first row with first 3 predictions, then predictions in rows of 4
        # First row has: Query + Pred0 + Pred1 + Pred2 (4 images)
        # Remaining rows have: 4 predictions each
        num_remaining = num_predictions - 3  # After first 3 predictions
        num_additional_rows = (num_remaining + 3) // 4  # How many full rows of 4
        num_rows = 1 + num_additional_rows
        
        total_width = 4 * IMG_HW + 3 * SPACE  # Always 4 columns
        total_height = num_rows * IMG_HW + (num_rows - 1) * SPACE
        concat_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))
        
        # First row: Query + first 3 predictions
        first_row_images = resized_images[:4]  # Query + first 3 predictions
        x = 0
        for img in first_row_images:
            concat_image.paste(img, (x, 0))
            x += IMG_HW + SPACE
        
        # Remaining rows: 4 predictions per row
        remaining_images = resized_images[4:]
        for row_idx in range(1, num_rows):
            row_images = remaining_images[(row_idx - 1) * 4:row_idx * 4]
            y_pos = row_idx * (IMG_HW + SPACE)
            x = 0
            for img in row_images:
                concat_image.paste(img, (x, y_pos))
                x += IMG_HW + SPACE
    else:
        # Single row layout
        total_h = len(resized_images) * IMG_HW + max(0, len(resized_images) - 1) * SPACE
        concat_image = Image.new('RGB', (total_h, IMG_HW), (255, 255, 255))
        x = 0
        for img in resized_images:
            concat_image.paste(img, (x, 0))
            x += IMG_HW + SPACE

    # Only add labels if label is not None
    if label is not None and len(labels) > 0:
        try:
            # Create labels for each row to match the grid layout
            if num_predictions > 4:
                # Grid layout: create label rows
                label_rows = []
                # First row: Query + first 3 predictions (4 labels)
                first_row_labels = labels[:4]
                label_rows.append(write_labels_to_image(first_row_labels))
                
                # Remaining rows: 4 predictions per row
                remaining_labels = labels[4:]
                for row_idx in range(1, num_rows):
                    row_labels = remaining_labels[(row_idx - 1) * 4:row_idx * 4]
                    # Pad with empty labels if needed to maintain 4 columns
                    while len(row_labels) < 4:
                        row_labels.append("")
                    if len(row_labels) > 0 and any(row_labels):  # Only add if there are labels
                        label_rows.append(write_labels_to_image(row_labels))
                
                # Concatenate all label rows vertically
                if label_rows:
                    labels_image = Image.fromarray(np.concatenate([np.array(lr) for lr in label_rows], axis=0))
                else:
                    labels_image = write_labels_to_image(labels[:4])
            else:
                labels_image = write_labels_to_image(labels)
            
            # Transform the images to np arrays for concatenation
            final_image = Image.fromarray(np.concatenate((np.array(labels_image), np.array(concat_image)), axis=0))
        except OSError:  # Handle error in case of missing PIL ImageFont
            final_image = concat_image
    else:
        # No labels, just return the images
        final_image = concat_image

    return final_image


def save_file_with_paths(query_path, preds_paths, output_path):
    file_content = []
    file_content.append("Query path:")
    file_content.append(query_path + "\n")
    file_content.append("Predictions paths:")
    file_content.append("\n".join(preds_paths) + "\n")
    with open(output_path, "w") as file:
        file.write("\n".join(file_content))


def save_preds_lightglue_only(predictions, match_counts, eval_ds, output_dir, lightglue_match_threshold=10):
    """Fast visualization for LightGlue-only mode using pre-computed match counts.
    This avoids re-running LightGlue matching which was already done during evaluation.
    
    Returns:
        results: List of dicts with query info and prediction details
    """
    viz_dir = output_dir / "preds"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for query_index, preds in enumerate(tqdm(predictions, desc="Saving LightGlue predictions")):
        query_path = eval_ds.queries_paths[query_index]
        list_of_images_paths = [query_path]
        
        # Add all predictions
        for pred in preds:
            pred_path = eval_ds.database_paths[pred]
            list_of_images_paths.append(pred_path)
        
        # Use pre-computed match counts to determine positive/negative
        preds_correct = [None]  # Query has no box
        num_matches_list = []
        prediction_details = []
        
        query_match_counts = match_counts[query_index]
        for pred_idx, (pred, num_matches) in enumerate(zip(preds, query_match_counts)):
            is_positive = bool(num_matches >= lightglue_match_threshold)  # Convert to native Python bool
            preds_correct.append(is_positive)
            num_matches_list.append(int(num_matches))  # Convert to int
            prediction_details.append({
                "pred_index": int(pred_idx),
                "image_path": str(eval_ds.database_paths[pred]),
                "is_positive": is_positive,
                "match_count": int(num_matches)
            })
        
        # Build and save visualization
        label = None
        prediction_image = build_prediction_image(
            list_of_images_paths, label, preds_correct, distances=None, num_matches=num_matches_list
        )
        
        # Save as JPG
        pred_image_path = viz_dir / f"{query_index:03d}.jpg"
        prediction_image.save(pred_image_path)
        
        # Save as PDF
        pred_pdf_path = viz_dir / f"{query_index:03d}.pdf"
        if prediction_image.mode != 'RGB':
            prediction_image = prediction_image.convert('RGB')
        prediction_image.save(pred_pdf_path, "PDF", resolution=100.0)
        
        # Save paths file
        save_file_with_paths(
            query_path=list_of_images_paths[0],
            preds_paths=list_of_images_paths[1:],
            output_path=viz_dir / f"{query_index:03d}.txt",
        )
        
        # Store result info with relative paths
        output_dir_name = output_dir.name if hasattr(output_dir, 'name') else str(output_dir).split('/')[-1]
        results.append({
            "query_index": int(query_index),
            "query_image": str(Path(query_path).name),  # Just filename for display
            "query_image_path": str(query_path),  # Full path for download
            "predictions": prediction_details,
            "visualization_image": f"{output_dir_name}/preds/{pred_image_path.name}"  # Relative path for serving
        })
    
    return results


def save_preds(predictions, eval_ds, output_dir, distances=None, distance_threshold=None,
                use_lightglue_only=False, lightglue_match_threshold=10, device=None):
    """For each query, save an image containing the query and its top predictions,
    and a file with the paths of the query and its predictions.
    Predictions are marked as positive (green) or negative (red) based on LightGlue matches and/or distance confidence.

    Parameters
    ----------
    predictions : np.array of shape [num_queries x num_preds_to_viz], with the preds
        for each query
    eval_ds : TestDataset
    output_dir : Path with the path to save the predictions
    distances : np.array of shape [num_queries x num_preds_to_viz], L2 distances for each prediction
        Lower distance = higher confidence (more similar)
    distance_threshold : float, optional
        Distance threshold for determining positive predictions. If None, uses adaptive threshold.
    use_lightglue_only : bool, default False
        If True, use only LightGlue matches to determine positive/negative (ignores distance threshold).
        If False, use LightGlue first, then distance threshold.
    lightglue_match_threshold : int, default 10
        Minimum number of LightGlue matches required for positive prediction (when use_lightglue_only=True).
    device : torch.device, optional
        Device to use for LightGlue. If None, defaults to cpu if cuda is not available.
    """
    viz_dir = output_dir / "preds"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize LightGlue extractor and matcher if available
    extractor = None
    matcher = None
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if LIGHTGLUE_AVAILABLE:
        extractor = SuperPoint(max_num_keypoints=256).eval().to(device)
        matcher = LightGlue(features='superpoint').eval().to(device)
    
    # Determine threshold if not provided
    if distances is not None and distance_threshold is None:
        # Use adaptive threshold: median of all first prediction distances
        first_pred_distances = distances[:, 0]
        distance_threshold = np.median(first_pred_distances) * 1.5  # 1.5x median as threshold
    
    total_processing_time = 0.0  # Track total processing time
    
    for query_index, preds in enumerate(tqdm(predictions, desc=f"Saving predictions")):
        query_path = eval_ds.queries_paths[query_index]
        list_of_images_paths = [query_path]
        
        # Add all predictions
        for pred in preds:
            pred_path = eval_ds.database_paths[pred]
            list_of_images_paths.append(pred_path)
        
        # Check each prediction with LightGlue first, then distance
        query_distances = None
        if distances is not None:
            query_distances = distances[query_index]
        
        preds_correct = [None]  # Query has no box
        num_matches_list = []  # Store number of matches for each prediction
        
        # Check each prediction
        for pred_idx, pred in enumerate(preds):
            pred_path = eval_ds.database_paths[pred]
            is_positive = True  # Default to positive
            num_matches = None  # Will store match count if available
            
            # First check: LightGlue matches
            if LIGHTGLUE_AVAILABLE and extractor is not None and matcher is not None:
                try:
                    start_time = time.time()
                    matches_result = light_glue_checker(
                        extractor, matcher, device, query_path, pred_path, score_threshold=0.85
                    )
                    processing_time = time.time() - start_time
                    total_processing_time += processing_time
                    num_matches = len(matches_result["matches"])
                    print(f"Query {query_index}, Pred {pred_idx}: {processing_time:.3f}s, matches: {num_matches}")
                    
                    if use_lightglue_only:
                        # LightGlue-only mode: use match threshold directly
                        is_positive = num_matches >= lightglue_match_threshold
                    else:
                        # Normal mode: LightGlue is first filter
                        if num_matches < 10:
                            # Less than 10 matches = negative (red)
                            is_positive = False
                except Exception as e:
                    # If LightGlue check fails
                    num_matches = None
                    print(f"Query {query_index}, Pred {pred_idx}: LightGlue check failed - {e}")
                    if use_lightglue_only:
                        # In LightGlue-only mode, if check fails, mark as negative
                        is_positive = False
            
            # Second check: Distance threshold (only if not using LightGlue-only mode and LightGlue didn't mark it as negative)
            if not use_lightglue_only:
                if is_positive and query_distances is not None and distance_threshold is not None:
                    dist = query_distances[pred_idx]
                    # Lower distance = higher confidence = positive (green)
                    # Higher distance = lower confidence = negative (red)
                    is_positive = dist < distance_threshold
                
                # Fallback: if no distance info, use heuristic
                if is_positive and query_distances is None:
                    num_positive = min(5, len(preds))
                    is_positive = pred_idx < num_positive
            
            preds_correct.append(is_positive)
            num_matches_list.append(num_matches)

        label = None  # Don't show labels, only images
        prediction_image = build_prediction_image(
            list_of_images_paths, label, preds_correct, distances=query_distances, num_matches=num_matches_list
        )
        
        # Save as JPG
        pred_image_path = viz_dir / f"{query_index:03d}.jpg"
        prediction_image.save(pred_image_path)
        
        # Save as PDF for zoom capability
        pred_pdf_path = viz_dir / f"{query_index:03d}.pdf"
        # Convert to RGB if needed (PDF requires RGB mode)
        if prediction_image.mode != 'RGB':
            prediction_image = prediction_image.convert('RGB')
        prediction_image.save(pred_pdf_path, "PDF", resolution=100.0)

        save_file_with_paths(
            query_path=list_of_images_paths[0],
            preds_paths=list_of_images_paths[1:],
            output_path=viz_dir / f"{query_index:03d}.txt",
        )
    
    # Print total processing time at the end
    if total_processing_time > 0:
        print(f"\nTotal processing time for all images: {total_processing_time:.3f}s ({total_processing_time/60:.2f} minutes)")
