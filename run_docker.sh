#!/bin/bash

# Script to build and run both backend and frontend Docker containers

set -e

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
docker stop image-similarity-backend image-similarity-frontend 2>/dev/null || true
docker rm image-similarity-backend image-similarity-frontend 2>/dev/null || true

# Create a network for the containers to communicate
echo -e "${YELLOW}Creating Docker network...${NC}"
docker network create image-similarity-network 2>/dev/null || true

# Run backend container
echo -e "${GREEN}Starting backend container...${NC}"
docker run -d \
    --name image-similarity-backend \
    --network image-similarity-network \
    -p 8000:8000 \
    image-similarity-backend

# Run frontend container
echo -e "${GREEN}Starting frontend container...${NC}"
docker run -d \
    --name image-similarity-frontend \
    --network image-similarity-network \
    -p 8080:8080 \
    image-similarity-frontend

echo -e "${GREEN}✓ Containers started successfully!${NC}"
echo -e "${BLUE}Backend API: http://localhost:8000${NC}"
echo -e "${BLUE}Frontend UI: http://localhost:8080${NC}"
echo -e "${BLUE}API Docs: http://localhost:8000/docs${NC}"
echo ""
echo -e "${YELLOW}To view logs:${NC}"
echo -e "  Backend:  docker logs -f image-similarity-backend"
echo -e "  Frontend: docker logs -f image-similarity-frontend"
echo ""
echo -e "${YELLOW}To stop containers:${NC}"
echo -e "  docker stop image-similarity-backend image-similarity-frontend"

