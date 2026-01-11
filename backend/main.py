from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import torch
import faiss
import os
import tempfile
import shutil
import zipfile
from glob import glob
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm

import visualizations
import vpr_models
from test_dataset import TestDataset, read_images_paths

app = FastAPI(title="VPR Methods Evaluation API")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Note: We'll use Form data for file uploads, so we don't need a Pydantic model for the request


class VPRResponse(BaseModel):
    success: bool
    message: str
    output_dir: str
    num_queries: int
    num_database: int


def validate_and_set_defaults(request):
    """Validate and set default values based on method"""
    method = request.method
    backbone = request.backbone
    descriptors_dimension = request.descriptors_dimension
    
    if method == "netvlad":
        if backbone not in [None, "VGG16"]:
            raise ValueError("When using NetVLAD the backbone must be None or VGG16")
        if descriptors_dimension not in [None, 4096, 32768]:
            raise ValueError("When using NetVLAD the descriptors_dimension must be one of [None, 4096, 32768]")
        if descriptors_dimension is None:
            request.descriptors_dimension = 4096

    elif method == "sfrs":
        if backbone not in [None, "VGG16"]:
            raise ValueError("When using SFRS the backbone must be None or VGG16")
        if descriptors_dimension not in [None, 4096]:
            raise ValueError("When using SFRS the descriptors_dimension must be one of [None, 4096]")
        if descriptors_dimension is None:
            request.descriptors_dimension = 4096

    elif method == "cosplace":
        if backbone is None:
            request.backbone = "ResNet50"
        if descriptors_dimension is None:
            request.descriptors_dimension = 2048
        if backbone == "VGG16" and descriptors_dimension not in [64, 128, 256, 512]:
            raise ValueError("When using CosPlace with VGG16 the descriptors_dimension must be in [64, 128, 256, 512]")
        if backbone == "ResNet18" and descriptors_dimension not in [32, 64, 128, 256, 512]:
            raise ValueError("When using CosPlace with ResNet18 the descriptors_dimension must be in [32, 64, 128, 256, 512]")
        if backbone in ["ResNet50", "ResNet101", "ResNet152"] and descriptors_dimension not in [32, 64, 128, 256, 512, 1024, 2048]:
            raise ValueError(f"When using CosPlace with {backbone} the descriptors_dimension must be in [32, 64, 128, 256, 512, 1024, 2048]")

    elif method == "convap":
        if backbone is None:
            request.backbone = "ResNet50"
        if descriptors_dimension is None:
            request.descriptors_dimension = 8192
        if backbone not in [None, "ResNet50"]:
            raise ValueError("When using Conv-AP the backbone must be None or ResNet50")
        if descriptors_dimension not in [None, 512, 2048, 4096, 8192]:
            raise ValueError("When using Conv-AP the descriptors_dimension must be one of [None, 512, 2048, 4096, 8192]")

    elif method == "mixvpr":
        if backbone is None:
            request.backbone = "ResNet50"
        if descriptors_dimension is None:
            request.descriptors_dimension = 4096
        if backbone not in [None, "ResNet50"]:
            raise ValueError("When using MixVPR the backbone must be None or ResNet50")
        if descriptors_dimension not in [None, 128, 512, 4096]:
            raise ValueError("When using MixVPR the descriptors_dimension must be one of [None, 128, 512, 4096]")

    elif method == "eigenplaces":
        if backbone is None:
            request.backbone = "ResNet50"
        if descriptors_dimension is None:
            request.descriptors_dimension = 2048
        if backbone == "VGG16" and descriptors_dimension not in [512]:
            raise ValueError("When using EigenPlaces with VGG16 the descriptors_dimension must be in [512]")
        if backbone == "ResNet18" and descriptors_dimension not in [256, 512]:
            raise ValueError("When using EigenPlaces with ResNet18 the descriptors_dimension must be in [256, 512]")
        if backbone in ["ResNet50", "ResNet101", "ResNet152"] and descriptors_dimension not in [128, 256, 512, 2048]:
            raise ValueError(f"When using EigenPlaces with {backbone} the descriptors_dimension must be in [128, 256, 512, 2048]")

    elif method == "eigenplaces-indoor":
        request.backbone = "ResNet50"
        request.descriptors_dimension = 2048

    elif method == "apgem":
        request.backbone = "Resnet101"
        request.descriptors_dimension = 2048

    elif method.startswith("anyloc"):
        request.backbone = "DINOv2"
        request.descriptors_dimension = 49152

    elif method == "salad":
        request.backbone = "DINOv2"
        request.descriptors_dimension = 8448

    elif method == "clique-mining":
        request.backbone = "DINOv2"
        request.descriptors_dimension = 8448

    elif method == "salad-indoor":
        request.backbone = "Dinov2"
        request.descriptors_dimension = 8448

    elif method == "cricavpr":
        request.backbone = "Dinov2"
        request.descriptors_dimension = 10752

    elif method == "megaloc":
        request.backbone = "Dinov2"
        request.descriptors_dimension = 8448

    elif method == "boq":
        if backbone not in [None, "ResNet50", "Dinov2"]:
            raise ValueError(f"When using BoQ the backbone must be ResNet50 or Dinov2")
        if backbone in [None, "ResNet50"]:
            request.backbone = "ResNet50"
            request.descriptors_dimension = 16384
            request.image_size = [384, 384]
        if backbone == "Dinov2":
            request.descriptors_dimension = 12288
            request.image_size = [322, 322]

    elif method == "dinomix":
        request.backbone = "Dinov2"
        request.descriptors_dimension = 4096
        request.image_size = [224, 224]

    elif method == "edtformer":
        request.backbone = "Dinov2"
        request.descriptors_dimension = 4096

    if request.image_size and len(request.image_size) > 2:
        raise ValueError(f"The image_size parameter can only take up to 2 values, but has received {len(request.image_size)}.")

    if max(request.recall_values) < request.num_preds_to_save:
        request.recall_values.append(request.num_preds_to_save)

    return request


def is_image_file(path: str) -> bool:
    """Check if path is an image file"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    return os.path.isfile(path) and os.path.splitext(path.lower())[1] in image_extensions


def get_image_paths(input_path: str) -> List[str]:
    """Get image paths from either a single image file or a folder"""
    if is_image_file(input_path):
        # Single image file
        return [input_path]
    elif os.path.isdir(input_path):
        # Folder - use existing logic to find images
        return read_images_paths(input_path)
    else:
        raise ValueError(f"Input path '{input_path}' is neither a valid image file nor a directory")


class CustomDataset:
    """Dataset class that works with provided image paths"""
    def __init__(self, database_paths, query_paths, image_size=None):
        self.database_paths = database_paths
        self.queries_paths = query_paths
        self.images_paths = list(database_paths) + list(query_paths)
        self.num_database = len(database_paths)
        self.num_queries = len(query_paths)
        
        import torchvision.transforms as transforms
        transformations = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if image_size:
            transformations.append(transforms.Resize(size=image_size, antialias=True))
        self.transform = transforms.Compose(transformations)
    
    def __getitem__(self, index):
        from PIL import Image
        image_path = self.images_paths[index]
        pil_img = Image.open(image_path).convert("RGB")
        normalized_img = self.transform(pil_img)
        return normalized_img, index
    
    def __len__(self):
        return len(self.images_paths)


async def save_uploaded_files(uploaded_files: List[UploadFile], temp_dir: Path) -> List[str]:
    """Save uploaded files to temporary directory and return their paths.
    Handles both individual image files and zip archives containing folders of images.
    """
    saved_paths = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    zip_extensions = {'.zip', '.tar', '.gz'}
    
    for idx, uploaded_file in enumerate(uploaded_files):
        filename = uploaded_file.filename or f"file_{idx}"
        file_ext = os.path.splitext(filename.lower())[1]
        
        file_path = temp_dir / filename
        with open(file_path, "wb") as f:
            content = await uploaded_file.read()
            f.write(content)
        
        # Check if it's a zip/archive file
        if file_ext in zip_extensions or file_ext == '.gz':
            # Extract zip file
            extract_dir = temp_dir / f"extracted_{idx}"
            extract_dir.mkdir()
            try:
                if file_ext == '.zip':
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                # Add support for other archive formats if needed
                
                # Find all images in extracted directory
                extracted_images = []
                for root, dirs, files in os.walk(extract_dir):
                    for file in files:
                        if os.path.splitext(file.lower())[1] in image_extensions:
                            extracted_images.append(os.path.join(root, file))
                
                if extracted_images:
                    saved_paths.extend(extracted_images)
                else:
                    raise ValueError(f"No images found in archive {filename}")
            except Exception as e:
                raise ValueError(f"Failed to extract archive {filename}: {str(e)}")
        elif file_ext in image_extensions:
            # It's an image file
            saved_paths.append(str(file_path))
        else:
            # Try to treat as image anyway
            saved_paths.append(str(file_path))
    
    return saved_paths


@app.post("/evaluate", response_model=VPRResponse)
async def evaluate_vpr(
    query_files: List[UploadFile] = File(..., description="Query image file(s) or zip file(s) containing folders of images"),
    database_files: List[UploadFile] = File(..., description="Database image file(s) or zip file(s) containing folders of images"),
    positive_dist_threshold: int = Form(default=25, description="Distance threshold for positive predictions"),
    num_workers: int = Form(default=4, description="Number of workers for data loading"),
    batch_size: int = Form(default=4, description="Batch size for processing"),
    log_dir: str = Form(default="default", description="Output directory name"),
    device: str = Form(default="cuda", description="Device to use (cuda or cpu)"),
    recall_values: str = Form(default="1,5,10,20", description="Recall values (comma-separated)"),
    image_size: Optional[str] = Form(default=None, description="Image size (comma-separated H,W or single int)"),
    save_descriptors: bool = Form(default=False, description="Whether to save descriptors"),
    distance_threshold: Optional[float] = Form(default=None, description="L2 distance threshold for positive predictions"),
    use_lightglue_only: bool = Form(default=False, description="If True, use only LightGlue matches to determine positive/negative (ignores distance threshold)"),
    lightglue_match_threshold: int = Form(default=10, description="Minimum number of LightGlue matches required for positive prediction (when use_lightglue_only=True)"),
):
    """
    Evaluate VPR method on uploaded images.
    Returns predictions and visualizations.
    """
    temp_dir = None
    try:
        # Create temporary directory for uploaded files
        temp_dir = Path(tempfile.mkdtemp())
        query_temp_dir = temp_dir / "queries"
        database_temp_dir = temp_dir / "database"
        query_temp_dir.mkdir()
        database_temp_dir.mkdir()
        
        # Save uploaded files
        query_paths = await save_uploaded_files(query_files, query_temp_dir)
        database_paths = await save_uploaded_files(database_files, database_temp_dir)
        
        # Fixed values (not user-configurable)
        method = "cosplace"
        backbone = "ResNet18"
        descriptors_dimension = 512
        
        # Parse recall_values from comma-separated string
        recall_values_list = [int(x.strip()) for x in recall_values.split(",")]
        
        # Parse image_size if provided
        image_size_list = None
        if image_size:
            image_size_list = [int(x.strip()) for x in image_size.split(",")]
        
        # Create a request-like object for validation
        class RequestObj:
            def __init__(self):
                self.method = method
                self.backbone = backbone
                self.descriptors_dimension = descriptors_dimension
                self.positive_dist_threshold = positive_dist_threshold
                self.num_workers = num_workers
                self.batch_size = batch_size
                self.log_dir = log_dir
                self.device = device
                self.recall_values = recall_values_list
                self.num_preds_to_save = 0  # Will be set later based on database size
                self.image_size = image_size_list
                self.save_descriptors = save_descriptors
                self.distance_threshold = distance_threshold
                self.use_lightglue_only = use_lightglue_only
                self.lightglue_match_threshold = lightglue_match_threshold
        
        request_obj = RequestObj()
        
        # Validate and set defaults
        request_obj = validate_and_set_defaults(request_obj)
        
        output_dir = Path("outputs") / request_obj.log_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        test_ds = CustomDataset(
            database_paths,
            query_paths,
            image_size=request_obj.image_size
        )
        
        # Set num_preds_to_save to number of database files
        request_obj.num_preds_to_save = len(database_paths)

        if request_obj.use_lightglue_only:
            # LightGlue-only mode: Skip VPR model, use only LightGlue matching
            from light_glue.lightglue.lightglue import LightGlue
            from light_glue.lightglue.disk import DISK
            from light_glue.run import light_glue_checker
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            extractor = DISK(max_num_keypoints=2048).eval().to(device)
            matcher = LightGlue(features='disk').eval().to(device)
            
            num_queries = len(query_paths)
            num_database = len(database_paths)
            num_preds = min(request_obj.num_preds_to_save, num_database)
            
            # For each query, match with all database images using LightGlue
            predictions = np.zeros((num_queries, num_preds), dtype=np.int64)
            
            print("Matching queries with database using LightGlue only...")
            for query_idx, query_path in enumerate(tqdm(query_paths, desc="LightGlue matching")):
                query_match_counts = []
                for db_idx, db_path in enumerate(database_paths):
                    try:
                        matches_result = light_glue_checker(
                            extractor, matcher, device, query_path, db_path, score_threshold=0.85
                        )
                        num_matches = len(matches_result["matches"])
                        query_match_counts.append((num_matches, db_idx))
                    except Exception as e:
                        print(f"Query {query_idx}, DB {db_idx}: LightGlue check failed - {e}")
                        query_match_counts.append((0, db_idx))
                
                # Sort by match count (descending) and take top num_preds
                query_match_counts.sort(reverse=True, key=lambda x: x[0])
                predictions[query_idx] = [db_idx for _, db_idx in query_match_counts[:num_preds]]
            
            # Create dummy distances (not used in LightGlue-only mode)
            distances = None
        else:
            # Normal VPR mode: Use VPR model + FAISS
            model = vpr_models.get_model(request_obj.method, request_obj.backbone, request_obj.descriptors_dimension)
            model = model.eval().to(request_obj.device)

            with torch.inference_mode():
                database_subset_ds = Subset(test_ds, list(range(test_ds.num_database)))
                database_dataloader = DataLoader(
                    dataset=database_subset_ds, num_workers=request_obj.num_workers, batch_size=request_obj.batch_size
                )
                all_descriptors = np.empty((len(test_ds), request_obj.descriptors_dimension), dtype="float32")
                for images, indices in tqdm(database_dataloader, desc="Extracting database descriptors"):
                    descriptors = model(images.to(request_obj.device))
                    descriptors = descriptors.cpu().numpy()
                    all_descriptors[indices.numpy(), :] = descriptors

                queries_subset_ds = Subset(
                    test_ds, list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries))
                )
                queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=request_obj.num_workers, batch_size=1)
                for images, indices in tqdm(queries_dataloader, desc="Extracting query descriptors"):
                    descriptors = model(images.to(request_obj.device))
                    descriptors = descriptors.cpu().numpy()
                    all_descriptors[indices.numpy(), :] = descriptors

            queries_descriptors = all_descriptors[test_ds.num_database :]
            database_descriptors = all_descriptors[: test_ds.num_database]

            if request_obj.save_descriptors:
                np.save(output_dir / "queries_descriptors.npy", queries_descriptors)
                np.save(output_dir / "database_descriptors.npy", database_descriptors)

            # Use a kNN to find predictions
            num_preds = request_obj.num_preds_to_save if request_obj.num_preds_to_save > 0 else max(request_obj.recall_values)
            # Limit num_preds to the number of database images available
            num_database = test_ds.num_database
            if num_preds > num_database:
                print(f"Warning: num_preds_to_save ({num_preds}) is larger than database size ({num_database}). Limiting to {num_database}.")
                num_preds = num_database
            
            faiss_index = faiss.IndexFlatL2(request_obj.descriptors_dimension)
            faiss_index.add(database_descriptors)
            del database_descriptors, all_descriptors

            # Capture distances (L2 distances) - lower distance = higher confidence
            distances, predictions = faiss_index.search(queries_descriptors, num_preds)

        # Save visualizations of predictions
        visualizations.save_preds(
            predictions, test_ds, output_dir, distances=distances, 
            distance_threshold=request_obj.distance_threshold,
            use_lightglue_only=request_obj.use_lightglue_only,
            lightglue_match_threshold=request_obj.lightglue_match_threshold
        )

        return VPRResponse(
            success=True,
            message="Evaluation completed successfully",
            output_dir=str(output_dir),
            num_queries=test_ds.num_queries,
            num_database=test_ds.num_database
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary files
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)


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
