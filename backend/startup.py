"""Startup event handler for preloading models"""
import torch
import vpr_models

# Global model cache - will be populated at startup
_model_cache = {}
_lightglue_extractor = None
_lightglue_matcher = None
_startup_complete = False  # Guard to ensure startup only runs once


def get_model_cache():
    """Get the model cache"""
    return _model_cache


def get_lightglue_models():
    """Get preloaded LightGlue models"""
    return _lightglue_extractor, _lightglue_matcher


async def startup_event():
    """Preload models at startup to avoid downloading during first request.
    This function is guarded to only run once, even if called multiple times.
    """
    global _startup_complete, _lightglue_extractor, _lightglue_matcher
    
    # Guard: only run once
    if _startup_complete:
        return
    
    print("=" * 60)
    print("Starting up - Preloading models...")
    print("=" * 60)
    
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")
    
    # Preload default VPR model
    print("\nPreloading default VPR model (cosplace, ResNet18, 512)...")
    try:
        default_model = vpr_models.get_model(
            method="cosplace",
            backbone="ResNet18",
            descriptors_dimension=512,
            device=device_str
        )
        default_model.eval()
        cache_key = f"cosplace_ResNet18_512_{device_str}"
        _model_cache[cache_key] = default_model
        print(f"✓ Default VPR model loaded successfully")
    except Exception as e:
        print(f"⚠ Warning: Failed to preload default VPR model: {e}")
        print("  Model will be loaded on first request")
    
    # Preload LightGlue models
    print("\nPreloading LightGlue models...")
    try:
        from light_glue.lightglue.lightglue import LightGlue
        from light_glue.lightglue.superpoint import SuperPoint
        
        _lightglue_extractor = SuperPoint(max_num_keypoints=256).eval().to(device)
        _lightglue_matcher = LightGlue(features='superpoint').eval().to(device)
        print(f"✓ LightGlue models loaded successfully")
    except Exception as e:
        print(f"⚠ Warning: Failed to preload LightGlue models: {e}")
        print("  Models will be loaded on first request")
    
    _startup_complete = True
    print("=" * 60)
    print("Startup complete - Ready to accept requests!")
    print("=" * 60)

