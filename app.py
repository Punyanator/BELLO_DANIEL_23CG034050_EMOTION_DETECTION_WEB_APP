from flask import Flask, render_template, request, jsonify, send_from_directory
import os, sqlite3, datetime, io, sys
import cv2, numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
processor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")

print("✅ Hugging Face model loaded successfully!")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "models", "emotion_model.h5")
DB_PATH = os.path.join(APP_DIR, "data", "users.db")
UPLOAD_FOLDER = os.path.join(APP_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Emotion labels expected from the model
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

app = Flask(__name__, static_folder='static', template_folder='templates')

# Initialize DB
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  image_path TEXT,
                  prediction TEXT,
                  confidence REAL,
                  timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()


# Fallback predictor returns neutral with confidence 1.0
def fallback_predict(face_img):
    return "neutral", 1.0

def predict_emotion(face_img):
    rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        conf, idx = torch.max(probs, dim=1)
        label = model.config.id2label[idx.item()]
        conf_value = round(min(conf.item(), 100.0), 2)
        return label, conf_value



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get('name', 'Anonymous')
    file = request.files.get('image')
    if not file:
        return jsonify(success=False, error="No image provided"), 400
    # Save upload
    ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    fname = f"{name}_{ts}.png"
    path = os.path.join(UPLOAD_FOLDER, fname)
    file.save(path)
    # Read image, detect largest face, crop
    img = cv2.imread(path)
    if img is None:
        return jsonify(success=False, error="Unable to read uploaded image"), 400
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        # store record with no face
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""INSERT INTO predictions (name, image_path, prediction, confidence, timestamp) VALUES (?,?,?,?,?)""", 
                  (name, path, "no_face", 0.0,datetime.datetime.now().isoformat()
))
        conn.commit()
        conn.close()
        return jsonify(success=True, prediction="no_face", confidence=0.0, face_image=None)
    # pick largest face
    x,y,w,h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
    face = img[y:y+h, x:x+w]
    label, conf = predict_emotion(face)
    # Save cropped face preview to uploads for frontend display
    face_preview_path = os.path.join(UPLOAD_FOLDER, f"crop_{fname}")
    cv2.imwrite(face_preview_path, face)
    # Store in DB
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""INSERT INTO predictions (name, image_path, prediction, confidence, timestamp) VALUES (?,?,?,?,?)""", 
              (name, path, label, float(conf), datetime.datetime.now().isoformat()
))
    conn.commit()
    conn.close()
    return jsonify(success=True, prediction=label, confidence=round(float(conf),3), face_image=os.path.basename(face_preview_path))

@app.route('/uploads/<path:filename>')
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Simple endpoint to check model presence
@app.route('/model_status')
def model_status():
    present = os.path.exists(MODEL_PATH)
    return jsonify(model_present=present, tensorflow_available=False)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 



