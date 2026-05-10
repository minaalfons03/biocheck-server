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
import json

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
        print("✓ Firebase connected successfully")
    except json.JSONDecodeError as e:
        print(f"✗ Firebase JSON parse error: {e}")
    except Exception as e:
        print(f"✗ Firebase init error: {e}")

init_firebase()

# ── In-memory face store ──────────────────────────────────────────────
known_faces = {}
FACES_DIR = "/tmp/known_faces"
os.makedirs(FACES_DIR, exist_ok=True)

# Reload faces saved to disk on startup
for fname in os.listdir(FACES_DIR):
    if fname.endswith(".jpg"):
        n = fname[:-4]
        known_faces[n] = os.path.join(FACES_DIR, fname)
        print(f"↺ Reloaded face: {n}")

# ── Routes ────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "registered_faces": list(known_faces.keys()),
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
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Resize large images to speed up processing
        max_size = 640
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.LANCZOS)

        img_path = os.path.join(FACES_DIR, f"{name}.jpg")
        img.save(img_path, quality=95)

        # Try multiple detector backends for better detection
        detected = False
        for backend in ["opencv", "ssd", "mtcnn"]:
            try:
                result = DeepFace.extract_faces(
                    img_path,
                    detector_backend=backend,
                    enforce_detection=True
                )
                if result:
                    detected = True
                    print(f"✓ Face detected with backend: {backend}")
                    break
            except Exception:
                continue

        if not detected:
            os.remove(img_path)
            return jsonify({"error": "No face found in image. Please use a clear, well-lit photo."}), 422

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

        # Resize for faster processing
        max_size = 640
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.LANCZOS)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name, quality=95)
            tmp_path = tmp.name

        result_name = "Unknown"
        best_distance = 1.0
        faces_found = 0

        # Try multiple backends to detect faces in the webcam frame
        for backend in ["opencv", "ssd", "mtcnn"]:
            try:
                faces = DeepFace.extract_faces(
                    tmp_path,
                    detector_backend=backend,
                    enforce_detection=False
                )
                found = [f for f in faces if f.get("confidence", 0) > 0.5]
                if found:
                    faces_found = len(found)
                    print(f"Faces detected ({backend}): {faces_found}")
                    break
            except Exception as e:
                print(f"Detection error ({backend}): {e}")
                continue

        if faces_found > 0:
            for name, face_path in known_faces.items():
                for backend in ["opencv", "ssd", "mtcnn"]:
                    try:
                        verify = DeepFace.verify(
                            tmp_path, face_path,
                            model_name="VGG-Face",
                            detector_backend=backend,
                            enforce_detection=False,
                            distance_metric="cosine"
                        )
                        dist = verify.get("distance", 1.0)
                        print(f"  → {name} ({backend}): distance={dist:.3f}")

                        # Relaxed threshold for better real-world matching
                        if dist < 0.5 and dist < best_distance:
                            best_distance = dist
                            result_name = name
                        break
                    except Exception as e:
                        print(f"Verify error for {name} ({backend}): {e}")
                        continue

        try:
            os.unlink(tmp_path)
        except:
            pass

        if result_name != "Unknown" and firebase_ref:
            firebase_ref.child("current_user").set(result_name)
            print(f"✓ Firebase updated: {result_name}")

        return jsonify({
            "name": result_name,
            "confidence": round(max(0, 1 - best_distance), 3),
            "faces_found": faces_found
        })

    except Exception as e:
        print(f"Recognise error: {e}")
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
