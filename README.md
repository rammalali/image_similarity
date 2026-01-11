# Image Similarity Evaluation

A web-based tool for evaluating Visual Place Recognition (VPR) methods using image similarity matching.

This project takes **one query image** and **one or multiple database images**, and finds images in the database that are similar to the query image.

![Example Result](./src/image.png)

In the visualization above:
- **Green bounding boxes** indicate **positive matches** (similar images)
- **Red bounding boxes** indicate **negative matches** (not similar images)

## Quick Start

To run both backend and frontend together:

```bash
./run_docker.sh
```

This will start:
- Backend API at `http://localhost:8000`
- Frontend UI at `http://localhost:8080`

## Individual Components

- [Backend Documentation](./backend/README.md) - How to run the backend Docker container only
- [Frontend Documentation](./frontend/README.md) - How to run the frontend Docker container only
