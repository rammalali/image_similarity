"""Evaluation logic for VPR methods"""
import numpy as np
import torch
import faiss
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import CustomDataset
from startup import get_model_cache, get_lightglue_models
import vpr_models


def get_device(device_str: str) -> torch.device:
    """Convert device string to torch.device, with fallback to CPU"""
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    return torch.device(device_str)


def _process_single_match(args):
    """Helper function to process a single query-database image pair"""
    extractor, matcher, device, query_path, db_path, db_idx, query_idx = args
    from light_glue.run import light_glue_checker
    
    try:
        matches_result = light_glue_checker(
            extractor, matcher, device, query_path, db_path, score_threshold=0.85
        )
        num_matches = len(matches_result["matches"])
        return (num_matches, db_idx, None)
    except Exception as e:
        return (0, db_idx, f"Query {query_idx}, DB {db_idx}: LightGlue check failed - {e}")


def run_lightglue_evaluation(query_paths, database_paths, device, num_preds, max_workers=None):
    """Run LightGlue-only evaluation with parallel processing
    
    Args:
        query_paths: List of query image paths
        database_paths: List of database image paths
        device: torch.device to use
        num_preds: Number of top predictions to return
        max_workers: Maximum number of parallel workers (None = auto-detect)
    
    Returns:
        predictions: np.array of shape [num_queries x num_preds]
        match_counts: np.array of shape [num_queries x num_preds] with match counts for each prediction
    """
    from light_glue.lightglue.lightglue import LightGlue
    from light_glue.lightglue.superpoint import SuperPoint
    
    extractor, matcher = get_lightglue_models()
    if extractor is None or matcher is None:
        extractor = SuperPoint(max_num_keypoints=256).eval().to(device)
        matcher = LightGlue(features='superpoint').eval().to(device)
    
    # Set max_workers based on device and available resources
    if max_workers is None:
        if device.type == "cuda":
            # For CUDA, use more workers since GPU can handle parallel requests
            max_workers = min(16, len(database_paths))
        else:
            # For CPU, use number of CPU cores
            import os
            max_workers = min(os.cpu_count() or 4, len(database_paths))
    
    num_queries = len(query_paths)
    predictions = np.zeros((num_queries, num_preds), dtype=np.int64)
    match_counts = np.zeros((num_queries, num_preds), dtype=np.int32)
    
    print(f"Matching queries with database using LightGlue only (parallel processing with {max_workers} workers)...")
    
    for query_idx, query_path in enumerate(tqdm(query_paths, desc="LightGlue matching")):
        query_match_counts = []
        
        # Prepare arguments for parallel processing
        match_args = [
            (extractor, matcher, device, query_path, db_path, db_idx, query_idx)
            for db_idx, db_path in enumerate(database_paths)
        ]
        
        # Process all database images for this query in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_db = {
                executor.submit(_process_single_match, args): args[5]  # db_idx
                for args in match_args
            }
            
            for future in as_completed(future_to_db):
                num_matches, db_idx, error_msg = future.result()
                if error_msg:
                    print(error_msg)
                query_match_counts.append((num_matches, db_idx))
        
        # Sort by match count and get top predictions
        query_match_counts.sort(reverse=True, key=lambda x: x[0])
        top_matches = query_match_counts[:num_preds]
        predictions[query_idx] = [db_idx for _, db_idx in top_matches]
        match_counts[query_idx] = [num_matches for num_matches, _ in top_matches]
    
    return predictions, match_counts


def run_vpr_evaluation(test_ds, request_obj, device):
    """Run normal VPR evaluation with model + FAISS"""
    model_cache = get_model_cache()
    device_str = "cuda" if device.type == "cuda" else "cpu"
    cache_key = f"{request_obj.method}_{request_obj.backbone}_{request_obj.descriptors_dimension}_{device_str}"
    
    if cache_key in model_cache:
        model = model_cache[cache_key]
    else:
        model = vpr_models.get_model(
            request_obj.method, request_obj.backbone, 
            request_obj.descriptors_dimension, device=device_str
        )
        model = model.eval().to(device)
        model_cache[cache_key] = model
    
    with torch.inference_mode():
        # Extract database descriptors
        database_subset_ds = Subset(test_ds, list(range(test_ds.num_database)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds, 
            num_workers=request_obj.num_workers, 
            batch_size=request_obj.batch_size
        )
        all_descriptors = np.empty((len(test_ds), request_obj.descriptors_dimension), dtype="float32")
        
        for images, indices in tqdm(database_dataloader, desc="Extracting database descriptors"):
            descriptors = model(images.to(device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors
        
        # Extract query descriptors
        queries_subset_ds = Subset(
            test_ds, list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries))
        )
        queries_dataloader = DataLoader(
            dataset=queries_subset_ds, 
            num_workers=request_obj.num_workers, 
            batch_size=1
        )
        for images, indices in tqdm(queries_dataloader, desc="Extracting query descriptors"):
            descriptors = model(images.to(device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors
    
    queries_descriptors = all_descriptors[test_ds.num_database:]
    database_descriptors = all_descriptors[:test_ds.num_database]
    
    # Save descriptors if requested
    if request_obj.save_descriptors:
        output_dir = Path("outputs") / request_obj.log_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "queries_descriptors.npy", queries_descriptors)
        np.save(output_dir / "database_descriptors.npy", database_descriptors)
    
    # FAISS search
    num_preds = request_obj.num_preds_to_save if request_obj.num_preds_to_save > 0 else max(request_obj.recall_values)
    num_database = test_ds.num_database
    if num_preds > num_database:
        print(f"Warning: num_preds_to_save ({num_preds}) is larger than database size ({num_database}). Limiting to {num_database}.")
        num_preds = num_database
    
    faiss_index = faiss.IndexFlatL2(request_obj.descriptors_dimension)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors
    
    distances, predictions = faiss_index.search(queries_descriptors, num_preds)
    return predictions, distances

