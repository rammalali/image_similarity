# Backend - VPR Evaluation API

FastAPI backend for VPR methods evaluation.

## Run with Docker

### Build the image

```bash
cd backend
docker build -t image-similarity-backend .
```

### Run the container

```bash
docker run -d \
    --name backend-container \
    --network image-similarity-network \
    --gpus all \
    -p 8000:8000 \
    image-similarity-backend
```

**Note:** The `--gpus all` flag enables GPU support. If you don't have GPU support configured, remove this flag to run in CPU mode.

### View logs

```bash
docker logs -f backend-container
```

### Stop the container

```bash
docker stop backend-container
docker rm backend-container
```

## API Documentation

Once running, visit:
- API Docs: `http://localhost:8000/docs`
- API Root: `http://localhost:8000`

