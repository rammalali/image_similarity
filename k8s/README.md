# Kubernetes Deployment Guide

This guide explains how to deploy the Image Similarity service on Kubernetes with multiple pods for improved performance.

## Performance Optimization Approaches

### 1. Parallel Processing Within Pods (Recommended)
**Status**: ✅ Already implemented in `evaluation.py`

This approach processes multiple database images in parallel within a single pod using `ThreadPoolExecutor`. It's:
- **Faster** for single requests (no network overhead)
- **Simpler** (no coordination needed)
- **More efficient** (better GPU utilization)

**When to use**: Always enabled by default. Works best for most use cases.

### 2. Multi-Pod Deployment (For Multiple Concurrent Users)
**Status**: ✅ K8s manifests provided

This approach runs multiple backend pods to handle multiple concurrent requests:
- **Better** for handling many users simultaneously
- **Provides** high availability and load balancing
- **Requires** multiple GPUs (one per pod) for optimal performance

**When to use**: When you have multiple users making requests at the same time.

## Deployment Steps

### Prerequisites
- Kubernetes cluster with GPU nodes (if using GPU)
- `kubectl` configured
- Docker images built and pushed to a registry

### 1. Build and Push Docker Images

```bash
# Build backend image
cd backend
docker build -t your-registry/image-similarity-backend:latest .
docker push your-registry/image-similarity-backend:latest

# Build frontend image
cd ../frontend
docker build -t your-registry/image-similarity-frontend:latest .
docker push your-registry/image-similarity-frontend:latest
```

### 2. Update Image References

Edit the deployment files to use your image registry:
- `backend-deployment.yaml` - Update `image:` field
- `frontend-deployment.yaml` - Update `image:` field

### 3. Configure GPU Support (if using GPU)

If your cluster has GPU nodes, uncomment the GPU-related sections in `backend-deployment.yaml`:

```yaml
resources:
  requests:
    nvidia.com/gpu: 1
  limits:
    nvidia.com/gpu: 1
# nodeSelector:
#   accelerator: nvidia-tesla-k80
# tolerations:
# - key: nvidia.com/gpu
#   operator: Exists
#   effect: NoSchedule
```

### 4. Deploy to Kubernetes

```bash
# Apply all manifests
kubectl apply -f backend-deployment.yaml
kubectl apply -f backend-service.yaml
kubectl apply -f frontend-deployment.yaml
kubectl apply -f frontend-service.yaml

# Check deployment status
kubectl get deployments
kubectl get pods
kubectl get services
```

### 5. Access the Services

**Option A: Port Forward (for testing)**
```bash
# Backend
kubectl port-forward service/backend-service 8040:8040

# Frontend
kubectl port-forward service/frontend-service 8080:8080
```

**Option B: LoadBalancer (for production)**
Change `type: ClusterIP` to `type: LoadBalancer` in the service files.

**Option C: Ingress (recommended for production)**
Set up an Ingress controller and create ingress rules.

## Configuration

### Environment Variables

You can configure the backend pods using environment variables in `backend-deployment.yaml`:

```yaml
env:
- name: USE_DISTRIBUTED_EVAL
  value: "false"  # Set to "true" to enable distributed mode (experimental)
- name: BACKEND_REPLICAS
  value: "3"      # Number of backend replicas
```

### Scaling

**Manual Scaling:**
```bash
kubectl scale deployment backend-deployment --replicas=5
```

**Auto-scaling (HPA):**
```bash
kubectl autoscale deployment backend-deployment --min=2 --max=10 --cpu-percent=80
```

## Performance Comparison

| Approach | Single Request Speed | Concurrent Requests | Complexity |
|----------|---------------------|---------------------|------------|
| Parallel Processing (Single Pod) | ⚡ Fast | Limited | Low |
| Multi-Pod (Multiple Users) | Same | ✅ High | Medium |
| Distributed (Single Request) | ⚠️ Slower (network overhead) | N/A | High |

**Recommendation**: Use parallel processing within pods (already implemented) for speed. Use multi-pod deployment for handling multiple concurrent users.

## Troubleshooting

### Check Pod Logs
```bash
kubectl logs -l app=image-similarity-backend
kubectl logs -l app=image-similarity-frontend
```

### Check GPU Allocation
```bash
kubectl describe pod <pod-name> | grep nvidia.com/gpu
```

### Check Service Endpoints
```bash
kubectl get endpoints backend-service
```

## Notes

- **Parallel processing within pods** is already implemented and enabled by default
- **Multi-pod deployment** is best for handling multiple concurrent users, not for speeding up a single request
- Each pod needs its own GPU for optimal performance
- Network overhead makes distributed processing across pods slower than parallel processing within a pod

