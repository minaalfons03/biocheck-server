import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import firebase_admin
from firebase_admin import credentials, db as firebase_db
from deepface import DeepFace
import tempfile

app = Flask(__name__)
CORS(app)

# ── Firebase ──────────────────────────────────────────────────────────
firebase_creds_json = os.environ.get("FIREBASE_CREDENTIALS")
firebase_ref = None
if firebase_creds_json:
    import json
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
    print("⚠ No FIREBASE_CREDENTIALS set")

# ── In-memory face store ──────────────────────────────────────────────
# { "name": "path/to/saved/image.jpg" }
known_faces = {}
FACES_DIR = "/tmp/known_faces"
os.makedirs(FACES_DIR, exist_ok=True)

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
        return jsonify({"error": "name and image required"}), 400
    try:
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        # Save image to disk so DeepFace can use it
        img_path = os.path.join(FACES_DIR, f"{name}.jpg")
        img.save(img_path)
        # Verify a face exists in the image
        result = DeepFace.extract_faces(img_path, detector_backend="opencv", enforce_detection=True)
        if not result:
            os.remove(img_path)
            return jsonify({"error": "No face found in image"}), 422
        known_faces[name] = img_path
        print(f"✓ Registered: {name}")
        return jsonify({"success": True, "name": name, "total": len(known_faces)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/recognise", methods=["POST"])
def recognise_face():
    data = request.get_json()
    image_b64 = data.get("image", "")
    if not image_b64:
        return jsonify({"error": "image required"}), 400
    if not known_faces:
        return jsonify({"name": None, "faces_found": 0, "message": "No faces registered yet"})
    try:
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        # Save snapshot temporarily
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name
        # Check each known face
        result_name = "Unknown"
        best_distance = 1.0
        faces_found = 0
        try:
            faces = DeepFace.extract_faces(tmp_path, detector_backend="opencv", enforce_detection=False)
            faces_found = len([f for f in faces if f.get("confidence", 0) > 0.5])
        except:
            faces_found = 0

        if faces_found > 0:
            for name, face_path in known_faces.items():
                try:
                    verify = DeepFace.verify(
                        tmp_path, face_path,
                        model_name="Facenet",
                        detector_backend="opencv",
                        enforce_detection=False
                    )
                    dist = verify.get("distance", 1.0)
                    if verify.get("verified") and dist < best_distance:
                        best_distance = dist
                        result_name = name
                except Exception as e:
                    print(f"Verify error for {name}: {e}")
                    continue

        os.unlink(tmp_path)

        if result_name != "Unknown" and firebase_ref:
            firebase_ref.child("current_user").set(result_name)

        confidence = round(max(0, 1 - best_distance), 3)
        return jsonify({"name": result_name, "confidence": confidence, "faces_found": faces_found})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/faces", methods=["GET"])
def list_faces():
    return jsonify({"faces": list(known_faces.keys())})

@app.route("/faces/<name>", methods=["DELETE"])
def delete_face(name):
    if name in known_faces:
        try:
            os.remove(known_faces[name])
        except:
            pass
        del known_faces[name]
        return jsonify({"success": True, "deleted": name})
    return jsonify({"error": "Not found"}), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
