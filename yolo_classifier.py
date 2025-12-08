'''
Train model for object detection using pretrained YOLO Detection Model

'''

from ultralytics import YOLO
<<<<<<< HEAD
#from dataset import PlayingCardDataset
=======
>>>>>>> d3ea4ba81214380317472d14d9070cdb50c51da5

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on the card dataset for 100 epochs
train_results = model.train(
    data="card.yaml",  # Path to dataset configuration file
<<<<<<< HEAD
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size for training
=======
    epochs=75,  # Number of training epochs
    imgsz=224,  # Image size for training
>>>>>>> d3ea4ba81214380317472d14d9070cdb50c51da5
    device="cpu",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)

# Evaluate the model's performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("data/test/ace of clubs/1.jpg")  # Predict on an image
results[0].show()  # Display results

# Export the model to ONNX format for deployment
path = model.export(format="onnx")  # Returns the path to the exported model