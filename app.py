from flask import Flask, render_template, request
import os
import re
import cv2
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ultralytics import YOLO

app = Flask(__name__)

# Paths
UPLOAD_FOLDER = "static/uploads"
PREDICTIONS_FOLDER = "static/predictions"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)

# Load YOLO Model for Multi-Plant Detection
yolo_model = YOLO("model/best.pt")

# Load plant data from CSV
plant_data = pd.read_csv("medicinal_plants.csv")

# Load disease-plant mapping CSV
disease_csv = pd.read_csv("disease_remedy.csv")


def predict_multi_plant(img_path):
    """Runs YOLO model on the image and saves the prediction"""
    results = yolo_model.predict(source=img_path, save=True, project=PREDICTIONS_FOLDER, name="results", exist_ok=True)
    
    # Get the latest saved image with predictions
    predicted_img_path = os.path.join(PREDICTIONS_FOLDER, "results", os.path.basename(img_path))
    
    return predicted_img_path


@app.route("/multi_plant_prediction", methods=["GET", "POST"])
def multi_plant_prediction():
    if "file" not in request.files:
        return render_template("res.html", error="No file uploaded")

    file = request.files["file"]
    if file.filename == "":
        return render_template("res.html", error="No file selected")

    # Save uploaded image
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    # Perform YOLO multi-plant prediction
    predicted_img_path = predict_multi_plant(img_path)

    return render_template(
        "res.html",
        uploaded_image=img_path,
        predicted_image=predicted_img_path,
    )


# Load trained model
model = tf.keras.models.load_model("model/Best_AyurPlantNet.keras")


# Load JSON from file
with open('class_labels.json', 'r') as f:
    data = json.load(f)

# Extract just the values into a list
class_labels = list(data.values())

def segment_image(img_path):
    """Segments the plant from the background using color thresholding"""
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Define color range for green plants
    lower_green = np.array([25, 40, 40], dtype=np.uint8)
    upper_green = np.array([85, 255, 255], dtype=np.uint8)

    # Create a mask to keep only the plant
    mask = cv2.inRange(hsv, lower_green, upper_green)
    segmented = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

    return img, segmented


def predict_segmented_image(img_path, model):
    """Segments an image, predicts its class, and fetches details from CSV."""
    original_img, segmented_img = segment_image(img_path)

    if original_img is None or segmented_img is None:
        print(f"Skipping prediction for {img_path} (Invalid Image)")
        return

    # Resize segmented image for model input
    resized_img = cv2.resize(segmented_img, (224, 224)) / 255.0
    resized_img = np.expand_dims(resized_img, axis=0)

    predictions = model.predict(resized_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_labels[predicted_class_index]
    confidence = np.max(predictions) * 100  

    plant_info = plant_data[plant_data["Plant"] == predicted_class]
    plant_details = plant_info.to_dict(orient="records")[0] if not plant_info.empty else None

    return original_img, segmented_img, predicted_class, confidence, plant_details

# Path to the static image folder
IMAGE_FOLDER = "static"

def clean_text(text):
    """Normalize the plant name by removing special characters and spaces"""
    text = text.lower().strip()  # Convert to lowercase and trim spaces
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)  # Remove special characters
    text = text.replace(" ", "") 
    return text

def find_image(plant_name):
    """Search for the plant image in the static folder"""
    cleaned_name = clean_text(plant_name).replace(" ", "_")  # Convert spaces to underscores
    image_extensions = [".jpg", ".jpeg", ".png", ".webp"]  # Common image formats
    
    for ext in image_extensions:
        image_path = f"{IMAGE_FOLDER}/{cleaned_name}{ext}"
        if os.path.exists(image_path):
            return f"/{image_path}"  # Return the relative path for Flask
    
    return "/static/not_found.jpg"  # Default image if not found

@app.route("/search", methods=["GET", "POST"])
def search_plant():
    plant_details = None  # Default to None
    if request.method == "POST":
        plant_name = request.form["plant_name"]
        cleaned_name = clean_text(plant_name)  # Normalize input
        
        # Search for plant name in CSV (case insensitive & clean)
        plant_info = plant_data[plant_data["Plant"].apply(clean_text) == cleaned_name]
        
        if not plant_info.empty:
            plant_details = plant_info.to_dict(orient="records")[0]  # Convert row to dictionary
            plant_details["image"] = find_image(cleaned_name)  # Add image path
        else:
            plant_details = {
                "Plant": "Not Found",
                "image": "/static/not_found.jpg",
                "Medicinal Uses": "Plant not found. Please try another name.",
                "Healing Properties": "Plant not found.",
                "Usage Method": "Plant not found.",
            }

    return render_template("search.html", plant_details=plant_details)


@app.route("/remedy", methods=["GET", "POST"])
def remedy():
    disease_list = disease_csv["Disease"].tolist()
    suggestions = []
    selected_disease = None

    if request.method == "POST":
        selected_disease = request.form["disease"]
        row = disease_csv[disease_csv["Disease"] == selected_disease]
        if not row.empty:
            plants = row.iloc[0]["Plants"].split("|")
            usages = row.iloc[0]["Usage"].split("|")

            # Create full detail including image
            for plant, usage in zip(plants, usages):
                plant_clean = clean_text(plant)
                image_path = find_image(plant_clean)
                suggestions.append({
                    "name": plant,
                    "usage": usage,
                    "image": image_path
                })

    return render_template("remedy.html", disease_list=disease_list, suggestions=suggestions, selected_disease=selected_disease)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/explore", methods=["GET"])
def explore():
    return render_template("explore.html")


@app.route("/result", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("result.html", error="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("result.html", error="No file selected")

        img_path = os.path.join("static/uploads", file.filename)
        file.save(img_path)

        original_img, segmented_img, predicted_class, confidence, plant_details = predict_segmented_image(img_path, model)

        if original_img is None:
            return render_template("result.html", error="Invalid image")

        segmented_img_path = os.path.join("static", "segmented_" + file.filename)
        cv2.imwrite(segmented_img_path, cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))

        return render_template(
            "result.html",
            uploaded_image=img_path,
            segmented_image=segmented_img_path,
            predicted_class=predicted_class,
            confidence=confidence,
            plant_details=plant_details,
        )

    return render_template("result.html")  # For GET request, just show form


if __name__ == "__main__":
    app.run(debug=True)
