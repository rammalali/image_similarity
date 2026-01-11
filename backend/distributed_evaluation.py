"""Distributed evaluation using multiple pods"""
import asyncio
import numpy as np
from typing import List, Tuple
import httpx
import os
from pathlib import Path


def get_backend_pods() -> List[str]:
    """Get list of backend pod URLs for distributed processing"""
    # Get service name from environment or use default
    service_name = os.getenv("BACKEND_SERVICE_NAME", "backend-service")
    namespace = os.getenv("KUBERNETES_NAMESPACE", "default")
    num_replicas = int(os.getenv("BACKEND_REPLICAS", "3"))
    
    # In K8s, we can use StatefulSet pod names or query the API
    # For simplicity, we'll use the service and let it load balance
    # In production, you'd query the K8s API to get individual pod IPs
    base_url = f"http://{service_name}.{namespace}.svc.cluster.local:8040"
    
    # For StatefulSet, pod names follow pattern: {statefulset-name}-{ordinal}
    # For Deployment, we'd need to query the K8s API
    # For now, return multiple URLs pointing to the same service
    # (In practice, you'd want to get individual pod IPs)
    return [base_url] * num_replicas


async def distribute_lightglue_evaluation(
    query_paths: List[str],
    database_paths: List[str],
    device: str,
    num_preds: int,
    use_distributed: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Distribute LightGlue evaluation across multiple pods using shared storage
    
    Note: This requires shared storage (PVC/NFS) accessible by all pods.
    For simpler setups, use local parallel processing instead.
    
    Args:
        query_paths: List of query image paths
        database_paths: List of database image paths
        device: Device string ("cuda" or "cpu")
        num_preds: Number of top predictions to return
        use_distributed: Whether to use distributed processing
    
    Returns:
        predictions: np.array of shape [num_queries x num_preds]
        match_counts: np.array of shape [num_queries x num_preds]
    """
    # For now, fall back to local parallel processing
    # True distributed processing would require:
    # 1. Shared storage for images
    # 2. Work queue (Redis/RabbitMQ) or direct pod communication
    # 3. Result aggregation
    
    # The parallel processing within a single pod (already implemented)
    # is usually faster than distributed processing due to:
    # - No network overhead
    # - No file transfer between pods
    # - Better GPU utilization within a single pod
    
    from evaluation import run_lightglue_evaluation
    from evaluation import get_device
    
    device_obj = get_device(device)
    # Use increased parallelism - the parallel processing is already very efficient
    # Run in thread pool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        run_lightglue_evaluation,
        query_paths, database_paths, device_obj, num_preds, 32
    )

