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
from firebase_admin import credentials, db as firebase_db, storage as firebase_storage

app = Flask(__name__)
CORS(app)

# ── Firebase ──────────────────────────────────────────────────────────
firebase_ref = None
storage_bucket = None

def init_firebase():
    global firebase_ref, storage_bucket
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
            'databaseURL': 'https://biocheckstation-default-rtdb.firebaseio.com/',
            'storageBucket': 'biocheckstation.appspot.com'
        })
        firebase_ref = firebase_db.reference('/')
        storage_bucket = firebase_storage.bucket()
        print("✓ Firebase connected (DB + Storage)")
    except Exception as e:
        print(f"✗ Firebase init error: {e}")

init_firebase()

# ── OpenCV ────────────────────────────────────────────────────────────
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

FACES_DIR = "/tmp/known_faces"
os.makedirs(FACES_DIR, exist_ok=True)

label_map = {}  # int -> name
name_map  = {}  # name -> int

def decode_image(b64_str):
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    if max(img.size) > 640:
        img.thumbnail((640, 640), Image.LANCZOS)
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

def extract_face(gray):
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return cv2.resize(gray[y:y+h, x:x+w], (200, 200))

def retrain():
    global recognizer, label_map, name_map
    faces_data, labels_data = [], []
    label_map, name_map = {}, {}
    counter = 0
    for fname in os.listdir(FACES_DIR):
        if not fname.endswith(".jpg"):
            continue
        name = fname[:-4]
        gray = cv2.imread(os.path.join(FACES_DIR, fname), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        face = extract_face(gray)
        if face is None:
            continue
        if name not in name_map:
            name_map[name] = counter
            label_map[counter] = name
            counter += 1
        faces_data.append(face)
        labels_data.append(name_map[name])
    if faces_data:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces_data, np.array(labels_data))
        print(f"✓ Trained on {len(faces_data)} face(s): {list(name_map.keys())}")
    else:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        print("⚠ No faces to train on")

def upload_face_to_firebase(name, img_path):
    """Upload face image to Firebase Storage for persistence."""
    if not storage_bucket:
        return
    try:
        blob = storage_bucket.blob(f"faces/{name}.jpg")
        blob.upload_from_filename(img_path, content_type='image/jpeg')
        print(f"✓ Uploaded {name} to Firebase Storage")
    except Exception as e:
        print(f"⚠ Storage upload failed: {e}")

def download_all_faces_from_firebase():
    """Download all face images from Firebase Storage on startup."""
    if not storage_bucket:
        print("⚠ No storage bucket — skipping face download")
        return
    try:
        blobs = list(storage_bucket.list_blobs(prefix="faces/"))
        if not blobs:
            print("⚠ No faces in Firebase Storage yet")
            return
        for blob in blobs:
            fname = blob.name.replace("faces/", "")
            if not fname.endswith(".jpg"):
                continue
            local_path = os.path.join(FACES_DIR, fname)
            blob.download_to_filename(local_path)
            print(f"↺ Downloaded face: {fname}")
    except Exception as e:
        print(f"⚠ Storage download failed: {e}")

def delete_face_from_firebase(name):
    """Delete face image from Firebase Storage."""
    if not storage_bucket:
        return
    try:
        blob = storage_bucket.blob(f"faces/{name}.jpg")
        blob.delete()
        print(f"✓ Deleted {name} from Firebase Storage")
    except Exception as e:
        print(f"⚠ Storage delete failed: {e}")

# ── Startup: load faces from Firebase Storage ─────────────────────────
print("↺ Loading faces from Firebase Storage...")
download_all_faces_from_firebase()
retrain()

# ── Routes ────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "registered_faces": list(name_map.keys()),
        "firebase": "connected" if firebase_ref else "disconnected",
        "storage": "connected" if storage_bucket else "disconnected"
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
        face = extract_face(gray)
        if face is None:
            return jsonify({"error": "No face detected. Use a clear, well-lit photo facing the camera."}), 422

        # Save to disk
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if max(img.size) > 640:
            img.thumbnail((640, 640), Image.LANCZOS)
        img_path = os.path.join(FACES_DIR, f"{name}.jpg")
        img.save(img_path, quality=95)

        # Upload to Firebase Storage for persistence across restarts
        upload_face_to_firebase(name, img_path)

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
        face = extract_face(gray)
        if face is None:
            return jsonify({"name": None, "faces_found": 0, "confidence": 0})

        label, distance = recognizer.predict(face)
        name = label_map.get(label, "Unknown")
        confidence = round(max(0, (100 - distance) / 100), 3)
        print(f"→ {name} | distance={distance:.1f} | confidence={confidence}")

        if distance > 100:
            name = "Unknown"
            confidence = 0

        if name != "Unknown" and firebase_ref:
            firebase_ref.child("current_user").set(name)
            print(f"✓ Firebase updated: {name}")

        return jsonify({"name": name, "confidence": confidence, "faces_found": 1})

    except Exception as e:
        print(f"Recognise error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/faces", methods=["GET"])
def list_faces():
    return jsonify({"faces": list(name_map.keys())})

@app.route("/faces/<name>", methods=["DELETE"])
def delete_face(name):
    if name in name_map:
        try:
            os.remove(os.path.join(FACES_DIR, f"{name}.jpg"))
        except:
            pass
        delete_face_from_firebase(name)
        retrain()
        return jsonify({"success": True, "deleted": name})
    return jsonify({"error": "Not found"}), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
