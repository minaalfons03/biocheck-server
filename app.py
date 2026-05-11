import os
import io
import base64
import json
import tempfile
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import cv2
import firebase_admin
from firebase_admin import credentials, db as firebase_db

app = Flask(__name__)
CORS(app)

# ── Firebase ──────────────────────────────────────────────────────────
firebase_ref = None

def init_firebase():
    global firebase_ref
    try:
        creds_json = os.environ.get("FIREBASE_CREDENTIALS", "")
        if not creds_json:
            print("⚠ FIREBASE_CREDENTIALS not set")
            return
        creds_json = creds_json.replace('\\n', '\n')
        creds_dict = json.loads(creds_json)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(creds_dict, f)
            creds_path = f.name
        cred = credentials.Certificate(creds_path)
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://biocheckstation-default-rtdb.firebaseio.com/'
        })
        firebase_ref = firebase_db.reference('/')
        print("✓ Firebase connected")
    except Exception as e:
        print(f"✗ Firebase init error: {e}")

init_firebase()

# ── OpenCV face detector & recognizer ────────────────────────────────
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
recognizer   = cv2.face.LBPHFaceRecognizer_create()

FACES_DIR = "/tmp/known_faces"
os.makedirs(FACES_DIR, exist_ok=True)

# Map: label_int -> name string
label_map = {}   # {0: "Mina", 1: "John", ...}
name_map  = {}   # {"Mina": 0, "John": 1}

def decode_image(b64_str):
    """Decode base64 image to OpenCV grayscale numpy array."""
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    # Resize large images
    max_size = 640
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return gray

def extract_face_region(gray):
    """Detect and return the largest face region from a grayscale image."""
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )
    if len(faces) == 0:
        return None, None
    # Pick the largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_region = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
    return face_region, (x, y, w, h)

def retrain():
    """Retrain the LBPH recognizer from all saved face images."""
    global recognizer, label_map, name_map
    faces_data = []
    labels_data = []
    label_map = {}
    name_map = {}
    label_counter = 0

    for fname in os.listdir(FACES_DIR):
        if not fname.endswith(".jpg"):
            continue
        name = fname[:-4]
        img_path = os.path.join(FACES_DIR, fname)
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        face_region, _ = extract_face_region(gray)
        if face_region is None:
            print(f"⚠ No face found in stored image for {name}, skipping")
            continue
        if name not in name_map:
            name_map[name] = label_counter
            label_map[label_counter] = name
            label_counter += 1
        faces_data.append(face_region)
        labels_data.append(name_map[name])

    if faces_data:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces_data, np.array(labels_data))
        print(f"✓ Recognizer trained on {len(faces_data)} face(s): {list(name_map.keys())}")
    else:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        print("⚠ No valid faces to train on")

# Train on startup from any saved images
retrain()

# ── Routes ────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "registered_faces": list(name_map.keys()),
        "firebase": "connected" if firebase_ref else "disconnected"
    })

@app.route("/register", methods=["POST"])
def register_face():
    data = request.get_json()
    name = data.get("name", "").strip()
    image_b64 = data.get("image", "")
    if not name or not image_b64:
        return jsonify({"error": "name and image required"}), 400
    try:
        gray = decode_image(image_b64)
        face_region, bbox = extract_face_region(gray)
        if face_region is None:
            return jsonify({"error": "No face detected. Use a clear, well-lit photo facing the camera."}), 422

        # Save original image to disk
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        max_size = 640
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.LANCZOS)
        img_path = os.path.join(FACES_DIR, f"{name}.jpg")
        img.save(img_path, quality=95)

        # Retrain recognizer with new face included
        retrain()

        print(f"✓ Registered: {name}")
        return jsonify({"success": True, "name": name, "total": len(name_map)})
    except Exception as e:
        print(f"Register error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/recognise", methods=["POST"])
def recognise_face():
    data = request.get_json()
    image_b64 = data.get("image", "")
    if not image_b64:
        return jsonify({"error": "image required"}), 400
    if not name_map:
        return jsonify({"name": None, "faces_found": 0, "message": "No faces registered yet"})
    try:
        gray = decode_image(image_b64)
        face_region, bbox = extract_face_region(gray)

        if face_region is None:
            return jsonify({"name": None, "faces_found": 0, "confidence": 0})

        label, distance = recognizer.predict(face_region)
        # LBPH distance: lower = better match. < 80 is a strong match, < 100 is acceptable
        confidence = max(0, round((100 - distance) / 100, 3))
        matched_name = label_map.get(label, "Unknown")

        print(f"→ Predicted: {matched_name} | distance={distance:.1f} | confidence={confidence}")

        if distance > 100:
            # Too uncertain, treat as unknown
            matched_name = "Unknown"
            confidence = 0

        if matched_name != "Unknown" and firebase_ref:
            firebase_ref.child("current_user").set(matched_name)
            print(f"✓ Firebase updated: {matched_name}")

        return jsonify({
            "name": matched_name,
            "confidence": confidence,
            "faces_found": 1
        })

    except Exception as e:
        print(f"Recognise error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/faces", methods=["GET"])
def list_faces():
    return jsonify({"faces": list(name_map.keys())})

@app.route("/faces/<name>", methods=["DELETE"])
def delete_face(name):
    if name in name_map:
        img_path = os.path.join(FACES_DIR, f"{name}.jpg")
        try:
            os.remove(img_path)
        except:
            pass
        retrain()  # retrain without deleted face
        return jsonify({"success": True, "deleted": name})
    return jsonify({"error": "Not found"}), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
