import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from predict_v5_bert import predict_complaint_v5

app = Flask(__name__)
CORS(app)

@app.route("/api/classify", methods=["POST"])
def classify():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data["text"]
    result = predict_complaint_v5(text)
    
    return jsonify(result)

@app.route("/api/download/<path:filename>", methods=["GET"])
def download_model(filename):
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_v5")
    return send_from_directory(models_dir, filename, as_attachment=True)

# ─── SERVE FRONTEND ────────────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:path>")
def serve_static(path):
    # Fallback to index.html if file not found (for SPA routing, though not strictly needed here)
    if os.path.exists(os.path.join(FRONTEND_DIR, path)):
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, "index.html")

if __name__ == "__main__":
    # Load model on startup
    print("Loading BERT Model...")
    predict_complaint_v5("test")
    print("Model loaded successfully. Starting server...")
    app.run(host="0.0.0.0", port=5000, debug=False)
