import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
CONFIG_PATH = BASE_DIR / "chatbot_config.txt"

try:
    SYSTEM_PROMPT = CONFIG_PATH.read_text(encoding="utf-8")
except OSError:
    SYSTEM_PROMPT = (
        "You are WildlifeGuard, an educational assistant focused on wildlife, "
        "biodiversity, conservation, and ecology. Politely redirect unrelated requests."
    )

client = genai.Client(api_key=API_KEY) if API_KEY else None


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/chat")
def chat():
    if not client:
        return jsonify({"error": "The Gemini API key is not configured."}), 503

    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Please enter a question."}), 400
    if len(message) > 6000:
        return jsonify({"error": "Please keep your question under 6,000 characters."}), 400

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.4,
                max_output_tokens=1200,
            ),
        )
        answer = (response.text or "").strip()
        if not answer:
            answer = "I couldn't create an answer for that. Please try rephrasing your wildlife or ecology question."
        return jsonify({"reply": answer})
    except Exception:
        app.logger.exception("Gemini request failed")
        return jsonify({"error": "WildlifeGuard couldn't reach Gemini right now. Please try again shortly."}), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
