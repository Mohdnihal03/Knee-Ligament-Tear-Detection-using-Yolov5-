import streamlit as st
import cv2
import numpy as np
import torch
import os 
# Class labels for DR severity
class_labels = ["No_DR", "mild", "moderate", "proliferate_DR", "severe"]

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model(model_path="yolo/best.pt"):
    print("Loading model from path:", model_path)
    """
    Loads the YOLOv5 model from the specified local path.

    Args:
        model_path (str, optional): Path to the YOLOv5 model file. Defaults to "best.pt".

    Returns:
        torch.nn.Module: The loaded YOLOv5 model.
    """
    model_name='best.pt'
    model = torch.hub.load(os.getcwd(), 'custom', source='local', path = model_name, force_reload = True)
    # model= 'yolo/best.pt'
    # model.load_state_dict(torch.load(model_path))
    # model.eval()

    return model
    

# Function to preprocess image
def preprocess_image(image, input_size=(416, 416)):
    """
    Preprocesses an image for model input.

    Args:
        image (np.ndarray): The image to preprocess.
        input_size (tuple, optional): The model's expected input size. Defaults to (416, 416).

    Returns:
        np.ndarray: The preprocessed image.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, input_size)
    image = np.transpose(image, (2, 0, 1))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to perform object detection
def detect_objects(model, image):
    """
    Performs object detection on an image using the model.

    Args:
        model (torch.nn.Module): The YOLOv5 model.
        image (np.ndarray): The preprocessed image.

    Returns:
        tuple: A tuple containing the processed image and detections (bounding boxes and class labels).
    """
    with torch.no_grad():
        results = model(image)
    detections = results.pandas().xyxy[0]  # Assuming single image inference
    return image, detections.to_numpy()

# Function to get detection information
def get_detection_info(detection):
    """
    Extracts bounding box coordinates, class label, and confidence score from a detection.

    Args:
        detection (dict): A dictionary containing detection information.

    Returns:
        tuple: A tuple containing bounding box coordinates (xmin, ymin, xmax, ymax), class label, and confidence score.
    """
    xmin, ymin, xmax, ymax = int(detection["xmin"]), int(detection["ymin"]), int(detection["xmax"]), int(detection["ymax"])
    class_id = int(detection["class"])
    class_label = class_labels[class_id]
    confidence_score = detection["conf"]
    return xmin, ymin, xmax, ymax, class_label, confidence_score

# Function to draw bounding box with label on an image
def draw_bounding_box(image, bbox, label):
    """
    Draws a bounding box with a label on an image.

    Args:
        image (np.ndarray): The image to draw on.
        bbox (tuple): A tuple containing bounding box coordinates (xmin, ymin, xmax, ymax).
        label (str): The class label to display.

    Returns:
        np.ndarray: The image with the bounding box and label drawn.
    """
    xmin, ymin, xmax, ymax = bbox
    color_map = [(0, 255, 0), (0, 165, 255), (0, 0, 255), (255, 0, 0), (128, 0, 128)]  # Color coding for severity levels
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color_map[class_labels.index(label)], 2)
    cv2.putText(image, f"{label}", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[class_labels.index(label)], 2)
    return image

# Streamlit app
def main():
    st.title("DR Severity Detection")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Load YOLOv5 model
        model = load_model()

        # Read image from file uploader
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

        # Preprocess image
        processed_image = preprocess_image(image)

        # Perform object detection
        processed_image, detections = detect_objects(model, processed_image)

        # Display detections on the original image
        for detection in detections:
            xmin, ymin, xmax, ymax, class_label, confidence_score = get_detection_info(detection)
            processed_image = draw_bounding_box(processed_image, (xmin, ymin, xmax, ymax), class_label)

        # Display original and processed image
        st.image([image, processed_image], caption=["Original Image", "Processed Image"], width=400)

if __name__ == "__main__":
    main()