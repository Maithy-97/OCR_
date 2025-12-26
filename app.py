import os
import cv2
import re
import numpy as np
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from paddleocr import PaddleOCR

# -------------------- LOGGING --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------- FLASK INIT --------------------
app = Flask(__name__)
CORS(app)  # allow all origins (restrict in production)

# -------------------- LOAD OCR MODEL (ONCE) --------------------
try:
    ocr = PaddleOCR(
        use_textline_orientation=True,  # updated param
        lang="en"
    )
    logger.info("PaddleOCR model loaded successfully")
except Exception as e:
    logger.exception("Failed to load PaddleOCR")
    raise e

# -------------------- UTILITY --------------------
def clean_text(text: str) -> str:
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    text = text.replace("O", "0").replace("I", "1")
    return text

def is_number_plate(text: str) -> bool:
    """
    Validate Indian vehicle registration number format:
    Format: XX00XX0000
    """
    pattern = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,2}[0-9]{4}$"
    return re.match(pattern, text) is not None

# -------------------- ROUTES --------------------
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "OCR API running"})

@app.route("/ocr", methods=["POST"])
def perform_ocr():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400

    file = request.files["file"]

    if not file or file.filename == "":
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    allowed_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    if not file.filename.lower().endswith(allowed_ext):
        return jsonify({"success": False, "error": "Unsupported file type"}), 400

    try:
        # Read image
        image_bytes = file.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"success": False, "error": "Invalid image data"}), 400

        # OCR
        result = ocr.predict(image)

        extracted_texts = []
        if result and result[0]:
            texts = result[0]["rec_texts"]
            scores = result[0]["rec_scores"]
            
            for text, score in zip(texts, scores):
                if score > 0.5:
                    cleaned = clean_text(text)
                    if is_number_plate(cleaned):
                        extracted_texts.append(cleaned)

        return jsonify({
            "success": True,
            "text": extracted_texts if extracted_texts else None
        })

    except Exception as e:
        logger.exception("OCR failed")
        return jsonify({"success": False, "error": "OCR processing failed"}), 500

# -------------------- SERVER ENTRYPOINT --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
