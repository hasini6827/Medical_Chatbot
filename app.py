from flask import Flask, request, jsonify, render_template
from openai import OpenAI
import config
app = Flask(__name__)
client =OpenAI(api_key = config.api_key)
chat_history =[]
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]
    response = client.responses.create(
        model = "gpt-4.1-mini",
        input = [
            {
                "role" : "system",
                "content":"""You are a helpful medical assistant chatbot.

Respond in a clear and friendly tone.
Use short sections with proper spacing.
Use headings like:

Possible Causes:
Questions:
Precautions:
When to Seek Help Immediately:

Do NOT use markdown symbols like ** or *.
Keep it clean and easy to read.
Do not provide a final diagnosis.
"""
            },
            {
                "role":"user",
                "content":user_message
            }
        ],
        max_output_tokens = 256
    )
    return jsonify({
        "reply":response.output_text

    })
@app.route("/")
def home():
    return render_template("index.html")
if __name__ == "__main__":
    app.run(debug=True)

