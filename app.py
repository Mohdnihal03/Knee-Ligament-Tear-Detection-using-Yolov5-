import cv2
import torch
import streamlit as st
import numpy as np
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Update these paths as needed
import os

model_path = os.path.join("C:/Users/nihall/Desktop/yolo/best.pt")    # Path to your exported YOLOv5 model
class_names = ["No_DR", "mild", "moderate", "proliferate_DR", "severe"]  # Class names for your dataset

def load_model():
  try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False)
    state_dict = torch.load(model_path)
    filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith('epoch') and not k.startswith('best_fitness')}
    model.load_state_dict(filtered_dict)
    return model
  except Exception as e:
    print(f"Error loading model: {e}")
    return None  # Or raise a specific exception


@st.cache  # Cache the model to avoid reloading on every run
def get_model():
  return load_model()

def predict(image):
  model = get_model()
  results = model(image)
  detections = results.pandas().xyxy[0]  # Get detections

  # Assuming your model predicts class index
  if detections.shape[0] > 0:
    class_id = int(detections["class"].iloc[0])
    class_name = class_names[class_id]
    return f"Predicted DR Severity: {class_name}"
  else:
    return "No DR detected in the image."

def main():
  st.title("DR Severity Prediction with YOLOv5")

  uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg" , "JFIF"])
  if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, channels="BGR")

    predictions = predict(image)
    st.success(predictions)

if __name__ == "__main__":
  main()
