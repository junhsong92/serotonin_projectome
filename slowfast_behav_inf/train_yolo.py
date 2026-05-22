import argparse
from ultralytics import YOLO

def train_yolo_model(data_yaml_path, model_variant, epochs, batch_size, img_size, project_name, run_name, save_period):
    """
    Fine-tunes a YOLOv8 model with specified parameters.

    Args:
        data_yaml_path (str): Path to the data.yaml file for the dataset.
        model_variant (str): The YOLOv8 model variant to use (e.g., 'yolov8n.pt', 'yolov8m.pt').
        epochs (int): The total number of training epochs.
        batch_size (int): The batch size for training.
        img_size (int): The image size for training (e.g., 640).
        project_name (str): The name of the project folder to save results.
        run_name (str): The specific name for this training run.
        save_period (int): Save checkpoint every N epochs.
    """
    # 1. Load a pre-trained YOLOv8 model
    try:
        model = YOLO(model_variant)
        print(f"@@@ Successfully loaded pre-trained model: {model_variant}")
    except Exception as e:
        print(f"### Error loading model '{model_variant}'. Please ensure the model name is correct and you have an internet connection for the initial download.")
        print(f"Error details: {e}")
        return

    # 2. Start model fine-tuning
    print("\nStarting model fine-tuning...")
    print(f" - Dataset: {data_yaml_path}")
    print(f" - Epochs: {epochs}")
    print(f" - Batch Size: {batch_size}")
    print(f" - Image Size: {img_size}")
    print(f" - Checkpoint Save Period: {save_period} epochs")
    
    try:
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            project=project_name,
            name=run_name,
            save_period=save_period,
            verbose=True # Ensure detailed logs are printed
        )
        
        print("\n@@@ Fine-tuning complete! The best model checkpoint has been saved.")
        print(f"Results and trained model weights are saved to: {results.save_dir}")

    except Exception as e:
        print(f"\n### An error occurred during training: {e}")
        print("Please check the path to your 'data.yaml' file and ensure your dataset is correctly formatted.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Fine-tune a YOLOv8 model on a custom dataset."
    )
    
    # Required argument
    parser.add_argument(
        "data_yaml", 
        type=str, 
        help="Path to the data.yaml file of your dataset."
    )

    # Optional arguments
    parser.add_argument(
        "-m", "--model", 
        type=str, 
        default="yolov8s.pt", 
        help="The base model to start fine-tuning from (e.g., 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt'). (default: yolov8s.pt)"
    )
    parser.add_argument(
        "-e", "--epochs", 
        type=int, 
        default=100, 
        help="Number of training epochs. (default: 100)"
    )
    parser.add_argument(
        "-b", "--batch_size", 
        type=int, 
        default=8, 
        help="Batch size for training. Adjust based on your VRAM. (default: 8)"
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=640, 
        help="Image size for training. (default: 640)"
    )
    parser.add_argument(
        "--project", 
        type=str, 
        default="yolo_finetune_results", 
        help="Project directory to save training results. (default: yolo_finetune_results)"
    )
    parser.add_argument(
        "--name", 
        type=str, 
        default="mouse_detection_run", 
        help="Specific name for this training run. (default: mouse_detection_run)"
    )
    parser.add_argument(
        "-sp", "--save_period",
        type=int,
        default=10,
        help="Save a checkpoint every N epochs. The best model is always saved. (default: 10)"
    )
    
    args = parser.parse_args()

    train_yolo_model(
        args.data_yaml, 
        args.model, 
        args.epochs, 
        args.batch_size, 
        args.imgsz, 
        args.project, 
        args.name,
        args.save_period
    )
