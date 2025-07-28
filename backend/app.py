from flask import Flask, request, jsonify
from chatbot import answer_question
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
@app.route("/", methods=["GET"])
def home():
    return "Backend is running!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"answer": "Please ask something."})

    try:
        answer = answer_question(query)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

