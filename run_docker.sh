#!/bin/bash

# Script to build and run both backend and frontend Docker containers

# Don't exit on error for GPU detection - we want to fall back to CPU
set +e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Building Docker images...${NC}"

# Build backend image
echo -e "${YELLOW}Building backend image...${NC}"
docker build -t image-similarity-backend ./backend

# Build frontend image
echo -e "${YELLOW}Building frontend image...${NC}"
docker build -t image-similarity-frontend ./frontend

# Stop and remove existing containers if they exist
echo -e "${YELLOW}Stopping existing containers (if any)...${NC}"
docker stop backend-container frontend-container 2>/dev/null || true
docker rm backend-container frontend-container 2>/dev/null || true

# Create a network for the containers to communicate
echo -e "${YELLOW}Creating Docker network...${NC}"
docker network create image-similarity-network 2>/dev/null || true

# Run backend container
echo -e "${GREEN}Starting backend container...${NC}"

# Check if GPU support is available
GPU_ARGS=""
if docker info 2>/dev/null | grep -q "nvidia"; then
    # Try a simple GPU test
    if timeout 5 docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
        GPU_ARGS="--gpus all"
        echo -e "${BLUE}✓ GPU support detected - using GPU${NC}"
    else
        echo -e "${YELLOW}⚠ GPU runtime detected but GPU access test failed. Running in CPU mode.${NC}"
        echo -e "${YELLOW}  To enable GPU: Install nvidia-container-toolkit and restart Docker${NC}"
    fi
else
    echo -e "${YELLOW}⚠ No GPU runtime detected. Running in CPU mode.${NC}"
    echo -e "${YELLOW}  To enable GPU: Install nvidia-container-toolkit${NC}"
fi

# Re-enable error checking for docker run
set -e
docker run -d \
    --name backend-container \
    --network image-similarity-network \
    $GPU_ARGS \
    -p 8000:8000 \
    image-similarity-backend

# Run frontend container
echo -e "${GREEN}Starting frontend container...${NC}"
docker run -d \
    --name frontend-container \
    --network image-similarity-network \
    -p 8080:8080 \
    image-similarity-frontend

echo -e "${GREEN}✓ Containers started successfully!${NC}"
echo -e "${BLUE}Backend API: http://localhost:8000${NC}"
echo -e "${BLUE}Frontend UI: http://localhost:8080${NC}"
echo -e "${BLUE}API Docs: http://localhost:8000/docs${NC}"
echo ""
echo -e "${YELLOW}To view logs:${NC}"
echo -e "  Backend:  docker logs -f backend-container"
echo -e "  Frontend: docker logs -f frontend-container"
echo ""
echo -e "${YELLOW}To stop containers:${NC}"
echo -e "  docker stop backend-container frontend-container"

