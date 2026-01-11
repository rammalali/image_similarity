"""FastAPI application for VPR Methods Evaluation API"""
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import tempfile
import shutil
import time
import torch
import zipfile
import io
import os
import visualizations
from utils import validate_and_set_defaults, CustomDataset, save_uploaded_files
from startup import startup_event, get_model_cache, get_lightglue_models
from evaluation import get_device, run_lightglue_evaluation, run_vpr_evaluation

app = FastAPI(title="VPR Methods Evaluation API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register startup event
app.on_event("startup")(startup_event)


class PredictionResult(BaseModel):
    query_index: int
    query_image: str
    predictions: List[dict]  # List of {image_path, is_positive, match_count, pred_index}
    visualization_image: str

class VPRResponse(BaseModel):
    success: bool
    message: str
    output_dir: str
    num_queries: int
    num_database: int
    results: Optional[List[PredictionResult]] = None


@app.post("/evaluate", response_model=VPRResponse)
async def evaluate_vpr(
    query_files: List[UploadFile] = File(...),
    database_files: List[UploadFile] = File(...),
    positive_dist_threshold: int = Form(default=25),
    num_workers: int = Form(default=4),
    batch_size: int = Form(default=4),
    log_dir: str = Form(default="default"),
    device: str = Form(default="cuda"),
    recall_values: str = Form(default="1,5,10,20"),
    image_size: Optional[str] = Form(default=None),
    save_descriptors: bool = Form(default=False),
    distance_threshold: Optional[float] = Form(default=None),
    use_lightglue_only: bool = Form(default=False),
    lightglue_match_threshold: int = Form(default=10),
):
    """Evaluate VPR method on uploaded images"""
    start_time = time.time()
    temp_dir = None
    try:
        # Setup temp directories
        temp_dir = Path(tempfile.mkdtemp())
        query_temp_dir = temp_dir / "queries"
        database_temp_dir = temp_dir / "database"
        query_temp_dir.mkdir()
        database_temp_dir.mkdir()
        
        # Save uploaded files
        query_paths = await save_uploaded_files(query_files, query_temp_dir)
        database_paths = await save_uploaded_files(database_files, database_temp_dir)
        
        # Create request object
        class RequestObj:
            def __init__(self):
                self.method = "cosplace"
                self.backbone = "ResNet18"
                self.descriptors_dimension = 512
                self.positive_dist_threshold = positive_dist_threshold
                self.num_workers = num_workers
                self.batch_size = batch_size
                self.log_dir = log_dir
                self.device = device
                self.recall_values = [int(x.strip()) for x in recall_values.split(",")]
                self.num_preds_to_save = len(database_paths)
                self.image_size = [int(x.strip()) for x in image_size.split(",")] if image_size else None
                self.save_descriptors = save_descriptors
                self.distance_threshold = distance_threshold
                self.use_lightglue_only = use_lightglue_only
                self.lightglue_match_threshold = lightglue_match_threshold
        
        request_obj = validate_and_set_defaults(RequestObj())
        output_dir = Path("outputs") / request_obj.log_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images to output directory for serving/downloading
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        queries_images_dir = images_dir / "queries"
        database_images_dir = images_dir / "database"
        queries_images_dir.mkdir(exist_ok=True)
        database_images_dir.mkdir(exist_ok=True)
        
        # Copy query images
        query_image_map = {}
        for idx, query_path in enumerate(query_paths):
            src_path = Path(query_path)
            dst_path = queries_images_dir / f"{idx:03d}_{src_path.name}"
            shutil.copy2(src_path, dst_path)
            query_image_map[query_path] = str(dst_path)
        
        # Copy database images
        database_image_map = {}
        for idx, db_path in enumerate(database_paths):
            src_path = Path(db_path)
            dst_path = database_images_dir / f"{idx:03d}_{src_path.name}"
            shutil.copy2(src_path, dst_path)
            database_image_map[db_path] = str(dst_path)
        
        # Create dataset
        test_ds = CustomDataset(database_paths, query_paths, image_size=request_obj.image_size)
        
        # Run evaluation
        device_obj = get_device(request_obj.device)
        if request_obj.use_lightglue_only:
            predictions, match_counts = run_lightglue_evaluation(
                query_paths, database_paths, device_obj, request_obj.num_preds_to_save
            )
            distances = match_counts  # Store match counts in distances for compatibility
        else:
            predictions, distances = run_vpr_evaluation(test_ds, request_obj, device_obj)


        # Save visualizations and get results
        results = None
        if request_obj.use_lightglue_only:
            # Fast path: use pre-computed match counts (no re-matching)
            results = visualizations.save_preds_lightglue_only(
                predictions, distances, test_ds, output_dir,
                lightglue_match_threshold=request_obj.lightglue_match_threshold
            )
            # Update image paths to use copied images
            for result in results:
                query_path = result.get("query_image_path", test_ds.queries_paths[result["query_index"]])
                result["query_image_path"] = str(query_image_map.get(query_path, query_path))
                for pred in result["predictions"]:
                    orig_path = pred["image_path"]
                    pred["image_path"] = str(database_image_map.get(orig_path, orig_path))
        else:
            # Normal path: may need to run LightGlue matching for visualization
            visualizations.save_preds(
                predictions, test_ds, output_dir, distances=distances,
                distance_threshold=request_obj.distance_threshold,
                use_lightglue_only=False,
                lightglue_match_threshold=request_obj.lightglue_match_threshold,
                device=device_obj
            )
            # For non-lightglue mode, create results from saved files
            results = []
            viz_dir = output_dir / "preds"
            for query_index in range(test_ds.num_queries):
                query_path = test_ds.queries_paths[query_index]
                preds = predictions[query_index]
                pred_details = []
                for pred_idx, pred in enumerate(preds):
                    orig_path = test_ds.database_paths[pred]
                    pred_details.append({
                        "pred_index": int(pred_idx),
                        "image_path": str(database_image_map.get(orig_path, orig_path)),
                        "is_positive": bool(True),  # Will be determined by reading visualization
                        "match_count": None
                    })
                output_dir_name = output_dir.name if hasattr(output_dir, 'name') else str(output_dir).split('/')[-1]
                results.append({
                    "query_index": int(query_index),
                    "query_image": str(Path(query_path).name),
                    "query_image_path": str(query_image_map.get(query_path, query_path)),
                    "predictions": pred_details,
                    "visualization_image": f"{output_dir_name}/preds/{query_index:03d}.jpg"
                })
        
        elapsed_time = time.time() - start_time
        print("=" * 60)
        print(f"Evaluation completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print("=" * 60)
        
        return VPRResponse(
            success=True,
            message="Evaluation completed successfully",
            output_dir=str(output_dir),
            num_queries=test_ds.num_queries,
            num_database=test_ds.num_database,
            results=results
        )
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Evaluation failed after {elapsed_time:.2f} seconds: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)


# Mount static files for serving images
outputs_path = Path("outputs")
outputs_path.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(outputs_path)), name="outputs")


@app.get("/results/{log_dir}/image/{filename:path}")
async def get_result_image(log_dir: str, filename: str):
    """Serve result images"""
    image_path = outputs_path / log_dir / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(image_path))


@app.post("/results/{log_dir}/download")
async def download_results(log_dir: str, results_data: dict):
    """Download results as zip file with positive/negative folders
    
    results_data should contain the results from the evaluation response
    """
    output_dir = outputs_path / log_dir
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="Results not found")
    
    preds_dir = output_dir / "preds"
    if not preds_dir.exists():
        raise HTTPException(status_code=404, detail="Predictions not found")
    
    # Create zip in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        results = results_data.get("results", [])
        
        for result in results:
            query_index = result["query_index"]
            query_image_path = Path(result.get("query_image_path", result.get("query_image", "")))
            
            # Copy query image to both folders (as reference)
            if query_image_path.exists():
                query_name = f"query_{query_index:03d}_{query_image_path.name}"
                zip_file.write(str(query_image_path), f"positive/{query_name}")
                zip_file.write(str(query_image_path), f"negative/{query_name}")
            
            # Copy prediction images to appropriate folders
            for pred in result["predictions"]:
                pred_image_path = Path(pred["image_path"])
                if pred_image_path.exists():
                    pred_name = f"pred_{query_index:03d}_{pred['pred_index']:03d}_{pred_image_path.name}"
                    if pred.get("is_positive", False):
                        zip_file.write(str(pred_image_path), f"positive/{pred_name}")
                    else:
                        zip_file.write(str(pred_image_path), f"negative/{pred_name}")
    
    zip_buffer.seek(0)
    
    return StreamingResponse(
        io.BytesIO(zip_buffer.read()),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=results_{log_dir}.zip"}
    )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "VPR Methods Evaluation API",
        "endpoints": {
            "/evaluate": "POST - Evaluate VPR method",
            "/docs": "GET - API documentation"
        }
    }
