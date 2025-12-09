''' 
Train model for object detection using pretrained YOLO Detection Model

'''
import torch
from ultralytics import YOLO

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    print("CUDA available:", torch.cuda.is_available())
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # Load a pretrained YOLO11n model
    model = YOLO("yolo11n.pt")

    # Train the model on the card dataset for 100 epochs
    train_results = model.train(
        data="data.yaml",  # Path to dataset configuration file
        epochs=100,  # Number of training epochs
        imgsz=416,
        batch=32,  # Image size for training
        device=0  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    )

    # Evaluate the model's performance on the validation set
    metrics = model.val()

    # Perform object detection on an image
    results = model("data/test/images/003587783_jpg.rf.1fde5478718d17e883ce43a158a376cb.jpg")  # Predict on an image
    results[0].show()  # Display results

    # Export the model to ONNX format for deployment
    path = model.export(format="onnx")  # Returns the path to the exported model