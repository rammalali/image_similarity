import parser
from pathlib import Path

import numpy as np
import torch
import faiss
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm

import visualizations
import vpr_models
from test_dataset import TestDataset


def main(args):
    output_dir = Path("outputs") / args.log_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model = vpr_models.get_model(args.method, args.backbone, args.descriptors_dimension)
    model = model.eval().to(args.device)

    test_ds = TestDataset(
        args.database_folder,
        args.queries_folder,
        positive_dist_threshold=args.positive_dist_threshold,
        image_size=args.image_size,
        use_labels=False,
    )

    with torch.inference_mode():
        database_subset_ds = Subset(test_ds, list(range(test_ds.num_database)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds, num_workers=args.num_workers, batch_size=args.batch_size
        )
        all_descriptors = np.empty((len(test_ds), args.descriptors_dimension), dtype="float32")
        for images, indices in tqdm(database_dataloader, desc="Extracting database descriptors"):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

        queries_subset_ds = Subset(
            test_ds, list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries))
        )
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers, batch_size=1)
        for images, indices in tqdm(queries_dataloader, desc="Extracting query descriptors"):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

    queries_descriptors = all_descriptors[test_ds.num_database :]
    database_descriptors = all_descriptors[: test_ds.num_database]

    if args.save_descriptors:
        np.save(output_dir / "queries_descriptors.npy", queries_descriptors)
        np.save(output_dir / "database_descriptors.npy", database_descriptors)

    # Use a kNN to find predictions
    num_preds = args.num_preds_to_save if args.num_preds_to_save > 0 else max(args.recall_values)
    # Limit num_preds to the number of database images available
    num_database = test_ds.num_database
    if num_preds > num_database:
        print(f"Warning: num_preds_to_save ({num_preds}) is larger than database size ({num_database}). Limiting to {num_database}.")
        num_preds = num_database
    
    faiss_index = faiss.IndexFlatL2(args.descriptors_dimension)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors

    # Capture distances (L2 distances) - lower distance = higher confidence
    distances, predictions = faiss_index.search(queries_descriptors, num_preds)

    # Save visualizations of predictions
    visualizations.save_preds(
        predictions, test_ds, output_dir, distances=distances, 
        distance_threshold=args.distance_threshold
    )


if __name__ == "__main__":
    args = parser.parse_arguments()
    main(args)
