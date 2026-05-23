from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("model/best_model.keras")

# =========================
# CLASS LABELS (A–J)
# IMPORTANT: must match training order
# =========================
class_names = ['A','B','C','D','E','F','G','H','I','J']

# =========================
# UPLOAD FOLDER
# =========================
UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# =========================
# HOME PAGE
# =========================
@app.route('/')
def home():
    return render_template("home.html")


# =========================
# PREDICTION PAGE
# =========================
@app.route('/predict-page')
def predict_page():
    return render_template("predict.html")


# =========================
# PREDICT FUNCTION
# =========================
@app.route('/predict', methods=['POST'])
def predict():

    file = request.files.get('file')

    if file and file.filename != '':

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # load image
        img = Image.open(filepath).convert("RGB")

        # preprocessing (MUST match training)
        img = img.resize((96, 96))
        img = np.array(img).astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # prediction
        prediction = model.predict(img)
        predicted_class = class_names[np.argmax(prediction)]

        return render_template(
            "predict.html",
            prediction=predicted_class,
            image_path=filepath
        )

    return render_template(
        "predict.html",
        prediction="No image uploaded"
    )


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)