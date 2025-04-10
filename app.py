from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os




# Initialize Flask app
app = Flask(__name__)

# Load the saved model
model = load_model("./mnist_cnn_model.h5")

# Function to preprocess the image
def preprocess_image(image_bytes):
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Grayscale
    image = image.resize((28, 28))
    img_array = np.array(image)

    # Invert and normalize
    img_array = 255 - img_array
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array

# API route
@app.route("/predict", methods=["POST"])
def predict():
    print("Request files:", request.files)  # 👈 Debug line
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty file name"}), 400

    try:
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)
        prediction = model.predict(processed_image)
        predicted_class = int(np.argmax(prediction))

        return jsonify({
            "predicted_digit": predicted_class,
            "confidence": float(np.max(prediction))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
