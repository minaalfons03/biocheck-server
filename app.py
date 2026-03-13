import os
import io
import base64
import numpy as np

# Force dlib to run without display
os.environ["DLIB_USE_CUDA"] = "0"
os.environ["DISPLAY"] = ""

import face_recognition
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import firebase_admin
from firebase_admin import credentials, db as firebase_db

app = Flask(__name__)
CORS(app)

# ── Firebase ──────────────────────────────────────────────────────────
firebase_creds_json = os.environ.get("FIREBASE_CREDENTIALS")
if firebase_creds_json:
    import json, tempfile
    creds_dict = json.loads(firebase_creds_json)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(creds_dict, f)
        creds_path = f.name
    cred = credentials.Certificate(creds_path)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://biocheckstation-default-rtdb.firebaseio.com/'
    })
    firebase_ref = firebase_db.reference('/')
    print("✓ Firebase connected")
else:
    firebase_ref = None
    print("⚠ No FIREBASE_CREDENTIALS — Firebase writes disabled")

# ── In-memory face store ──────────────────────────────────────────────
known_faces = {}  # { "name": np.array(encoding) }

# ── Routes ────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "online", "registered_faces": list(known_faces.keys())})

@app.route("/register", methods=["POST"])
def register_face():
    data = request.get_json()
    name = data.get("name", "").strip()
    image_b64 = data.get("image", "")
    if not name or not image_b64:
        return jsonify({"error": "name and image are required"}), 400
    try:
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)
        encodings = face_recognition.face_encodings(img_np)
        if not encodings:
            return jsonify({"error": "No face found in the image"}), 422
        known_faces[name] = encodings[0]
        print(f"✓ Registered: {name}")
        return jsonify({"success": True, "name": name, "total": len(known_faces)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/recognise", methods=["POST"])
def recognise_face():
    data = request.get_json()
    image_b64 = data.get("image", "")
    if not image_b64:
        return jsonify({"error": "image is required"}), 400
    try:
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)
        face_locations = face_recognition.face_locations(img_np)
        face_encodings = face_recognition.face_encodings(img_np, face_locations)
        if not face_encodings:
            return jsonify({"name": None, "faces_found": 0, "message": "No face detected"})
        result_name = "Unknown"
        best_distance = 1.0
        for encoding in face_encodings:
            if not known_faces:
                break
            known_names = list(known_faces.keys())
            known_encodings = list(known_faces.values())
            distances = face_recognition.face_distance(known_encodings, encoding)
            best_idx = int(np.argmin(distances))
            dist = float(distances[best_idx])
            if dist < 0.50 and dist < best_distance:
                best_distance = dist
                result_name = known_names[best_idx]
        if result_name != "Unknown" and firebase_ref:
            firebase_ref.child("current_user").set(result_name)
        return jsonify({"name": result_name, "confidence": round(1 - best_distance, 3), "faces_found": len(face_encodings)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/faces", methods=["GET"])
def list_faces():
    return jsonify({"faces": list(known_faces.keys())})

@app.route("/faces/<name>", methods=["DELETE"])
def delete_face(name):
    if name in known_faces:
        del known_faces[name]
        return jsonify({"success": True, "deleted": name})
    return jsonify({"error": "Not found"}), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
