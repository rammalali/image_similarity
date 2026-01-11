# Frontend - VPR Evaluation UI

Frontend web interface for the VPR Evaluation Tool.

## Run with Docker

### Build the image

```bash
cd frontend
docker build -t image-similarity-frontend .
```

### Run the container

```bash
docker run -d \
    --name frontend-container \
    --network image-similarity-network \
    -p 8080:8080 \
    image-similarity-frontend
```

### View logs

```bash
docker logs -f frontend-container
```

### Stop the container

```bash
docker stop frontend-container
docker rm frontend-container
```

## Access the UI

Once running, visit: `http://localhost:8080`
