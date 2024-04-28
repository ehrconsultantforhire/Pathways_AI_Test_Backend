from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from src.crew import run_crew

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route("/api/ai", methods=["POST"])
def handle_route():
    data = request.json

    if not data or "prompt" not in data:
        return jsonify({"response": "Prompt not available", "success": False}), 400

    prompt = data["prompt"]

    if prompt:
        processed_text = run_crew(prompt)
        f = open("output.md", "w")
        f.write(processed_text)
        f.close()
        return jsonify({"response": processed_text, "success": True}), 200
    else:
        # Handle missing prompt error
        return jsonify({"response": None, "success": False}), 400


if __name__ == "__main__":
    app.run(debug=True, port=8000)
