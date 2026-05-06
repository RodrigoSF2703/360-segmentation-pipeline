from src.training.dataset_builder import YOLODatasetBuilder


def main():
    builder = YOLODatasetBuilder(
        images_dir="data/processed/images",
        masks_dir="data/processed/masks",
        output_dir="data/processed/yolo_dataset",
        train_ratio=0.8,
        min_contour_area=200,
    )

    builder.build()


if __name__ == "__main__":
    main()