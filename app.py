import streamlit as st
import cv2
import numpy as np
import time
from ultralytics import YOLO

# Title and description
st.title("Intelligent Traffic Monitoring System")
st.write("This application detects and tracks vehicles on highways in real-time using the YOLO model. It can process videos, images, or a live camera feed to count vehicles.")

# Sidebar for user input
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
source_type = st.sidebar.radio("Select Source", ("Image", "Video", "Webcam", "IP Camera"))

# Load YOLO model (fixed to YOLOv8n)
model = YOLO("yolov8n.pt")

# Vehicle classes to count
vehicle_classes = ["car", "truck", "bus", "motorbike"]

# Function to detect and count vehicles
def process_frame(frame):
    results = model.predict(frame)
    vehicle_count = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]

            # Filter for vehicle classes only
            if conf >= confidence_threshold and label in vehicle_classes:
                vehicle_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"{label.capitalize()}: {conf:.2f}"
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame, vehicle_count

# Display count in large text
def display_count(container, count):
    container.markdown(f"<h2 style='text-align: center; color: green;'>Detected Vehicles: {count}</h2>", unsafe_allow_html=True)

# Image processing
if source_type == "Image":
    uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
        processed_image, vehicle_count = process_frame(image)
        st.image(processed_image, channels="BGR", caption="Processed Image")
        display_count(st, vehicle_count)

# Video processing
elif source_type == "Video":
    uploaded_video = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_video is not None:
        st.video(uploaded_video)
        st.write("Processing video for vehicle detection and tracking...")

        # Save the uploaded video temporarily
        video_bytes = uploaded_video.read()
        with open("temp_video.mp4", "wb") as f:
            f.write(video_bytes)

        # Open the video file
        cap = cv2.VideoCapture("temp_video.mp4")
        stframe = st.empty()
        count_container = st.empty()

        # Process each frame in the video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame for faster processing
            frame = cv2.resize(frame, (640, 360))  # Resize to 640x360 for faster processing
            
            # Detect vehicles and get the count
            frame, vehicle_count = process_frame(frame)

            # Display the processed frame
            stframe.image(frame, channels="BGR")
            display_count(count_container, vehicle_count)

            # Adjust sleep time for smoother playback
            time.sleep(0.01)  # Adjust this to balance speed and smoothness

        cap.release()

# Webcam or IP camera stream
elif source_type in ["Webcam", "IP Camera"]:
    if source_type == "IP Camera":
        ip_address = st.sidebar.text_input("Enter IP Camera URL", "http://<IP>:<PORT>/video")
    else:
        ip_address = 0  # For webcam

    if st.sidebar.button("Start Stream"):
        cap = cv2.VideoCapture(ip_address)
        if not cap.isOpened():
            st.error("Error: Unable to open video stream.")
        else:
            stframe = st.empty()
            count_container = st.empty()  # Container for vehicle count to update on the same line

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Error: Unable to read frame.")
                    break

                # Resize frame for faster processing
                frame = cv2.resize(frame, (640, 360))

                # Detect vehicles and get the count
                frame, vehicle_count = process_frame(frame)

                # Display the processed frame
                stframe.image(frame, channels="BGR")
                display_count(count_container, vehicle_count)

                # Stop stream if user stops app
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()

# Close any open windows if needed
cv2.destroyAllWindows()
