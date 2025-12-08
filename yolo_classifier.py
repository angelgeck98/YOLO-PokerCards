''' 
Train model for object detection using pretrained YOLO Detection Model

'''

from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on the card dataset for 100 epochs
train_results = model.train(
    data="data.yaml",  # Path to dataset configuration file
    epochs=50,  # Number of training epochs
    imgsz=416,  # Image size for training
    device="cpu",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)

# Evaluate the model's performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("data/test/images/003587783_jpg.rf.1fde5478718d17e883ce43a158a376cb.jpg")  # Predict on an image
results[0].show()  # Display results

# Export the model to ONNX format for deployment
path = model.export(format="onnx")  # Returns the path to the exported model