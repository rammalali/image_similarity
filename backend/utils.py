"""Utility functions for file handling, validation, and datasets"""
import os
import zipfile
from pathlib import Path
from typing import List
from fastapi import UploadFile
import torchvision.transforms as transforms
from PIL import Image
from test_dataset import read_images_paths


def validate_and_set_defaults(request):
    """Validate and set default values based on method"""
    method = request.method
    backbone = request.backbone
    descriptors_dimension = request.descriptors_dimension
    
    defaults = {
        "netvlad": {"backbone": None, "descriptors_dimension": 4096},
        "sfrs": {"backbone": None, "descriptors_dimension": 4096},
        "cosplace": {"backbone": "ResNet50", "descriptors_dimension": 2048},
        "convap": {"backbone": "ResNet50", "descriptors_dimension": 8192},
        "mixvpr": {"backbone": "ResNet50", "descriptors_dimension": 4096},
        "eigenplaces": {"backbone": "ResNet50", "descriptors_dimension": 2048},
        "eigenplaces-indoor": {"backbone": "ResNet50", "descriptors_dimension": 2048},
        "apgem": {"backbone": "Resnet101", "descriptors_dimension": 2048},
        "anyloc": {"backbone": "DINOv2", "descriptors_dimension": 49152},
        "salad": {"backbone": "DINOv2", "descriptors_dimension": 8448},
        "clique-mining": {"backbone": "DINOv2", "descriptors_dimension": 8448},
        "salad-indoor": {"backbone": "Dinov2", "descriptors_dimension": 8448},
        "cricavpr": {"backbone": "Dinov2", "descriptors_dimension": 10752},
        "megaloc": {"backbone": "Dinov2", "descriptors_dimension": 8448},
        "boq": {"backbone": "ResNet50", "descriptors_dimension": 16384, "image_size": [384, 384]},
        "dinomix": {"backbone": "Dinov2", "descriptors_dimension": 4096, "image_size": [224, 224]},
        "edtformer": {"backbone": "Dinov2", "descriptors_dimension": 4096},
    }
    
    # Handle method-specific defaults
    if method in defaults:
        config = defaults[method]
        if backbone is None:
            request.backbone = config["backbone"]
        if descriptors_dimension is None:
            request.descriptors_dimension = config["descriptors_dimension"]
        if "image_size" in config and request.image_size is None:
            request.image_size = config["image_size"]
    
    # Validation
    if method == "boq" and backbone not in [None, "ResNet50", "Dinov2"]:
        raise ValueError(f"When using BoQ the backbone must be ResNet50 or Dinov2")
    
    if request.image_size and len(request.image_size) > 2:
        raise ValueError(f"The image_size parameter can only take up to 2 values, but has received {len(request.image_size)}.")
    
    if max(request.recall_values) < request.num_preds_to_save:
        request.recall_values.append(request.num_preds_to_save)
    
    return request


class CustomDataset:
    """Dataset class that works with provided image paths"""
    def __init__(self, database_paths, query_paths, image_size=None):
        self.database_paths = database_paths
        self.queries_paths = query_paths
        self.images_paths = list(database_paths) + list(query_paths)
        self.num_database = len(database_paths)
        self.num_queries = len(query_paths)
        
        transformations = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if image_size:
            transformations.append(transforms.Resize(size=image_size, antialias=True))
        self.transform = transforms.Compose(transformations)
    
    def __getitem__(self, index):
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
        
        if file_ext in zip_extensions:
            extract_dir = temp_dir / f"extracted_{idx}"
            extract_dir.mkdir()
            try:
                if file_ext == '.zip':
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                
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
            saved_paths.append(str(file_path))
        else:
            saved_paths.append(str(file_path))
    
    return saved_paths

