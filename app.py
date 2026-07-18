from flask import Flask, request, jsonify, render_template, session
from openai import OpenAI
import json
import os

try:
    import config
    api_key = config.api_key
except ImportError:
    api_key = os.environ.get("OPENAI_API_KEY")

app = Flask(__name__)
app.secret_key = "change_this_to_a_random_secret_key"

client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = """
You are MedAssist, an intelligent AI medical assistant.

Your personality:
- Speak naturally like ChatGPT or August AI.
- Be warm, calm, empathetic and conversational.
- Never sound robotic.
- Keep replies short (50-120 words unless necessary).
- Avoid long bullet lists.
- Ask only ONE follow-up question when required.
- If enough information is available, stop asking questions and provide guidance.
- Never ask the same question twice.
- Remember everything the user has already confirmed.
- If the user answers with short replies like "yes", "no", "itchy", "left eye", "2 days", interpret them as answers to your previous question.
- Never repeat questions about already confirmed symptoms.
- Give only the 2-3 most likely possibilities.
- Never diagnose with certainty.
- Recommend emergency care only when appropriate.

You will also receive a JSON object called KNOWN_FACTS.

KNOWN_FACTS contains information already confirmed during the conversation.

Rules:

1. Always read KNOWN_FACTS before replying.
2. Never ask about anything already present in KNOWN_FACTS.
3. Update your reasoning using KNOWN_FACTS.
4. Ask only the next most important unanswered question.
5. Once enough information is collected, provide your assessment.

Reply naturally.
Conversation Strategy

1. First understand the user's concern.

2. If enough information is available, answer immediately.

3. Only ask a follow-up question when it will significantly improve your advice.

4. Ask only ONE question at a time.

5. Never ask for information already present in KNOWN_FACTS.

6. Usually stop after collecting 2–3 important pieces of information.

7. Once enough information is collected:
   - Explain the likely causes.
   - Give concise advice.
   - Mention warning signs only if relevant.

8. Do not continue asking questions just to keep the conversation going.

9. Avoid large paragraphs and long lists.

10. Sound like ChatGPT or August AI:
    - friendly
    - concise
    - conversational
    - reassuring
"""


def initialize_session():
    if "chat_history" not in session:
        session["chat_history"] = []

    if "known_facts" not in session:
        session["known_facts"] = {}


def update_known_facts(history, facts):
    """
    Uses GPT to update symptom memory.
    """

    conversation = ""

    for msg in history[-10:]:
        role = msg["role"].capitalize()
        conversation += f"{role}: {msg['content']}\n"

    prompt = f"""
Current known facts:

{json.dumps(facts, indent=2)}

Conversation:

{conversation}

Update the known facts.

Rules:

- Return ONLY valid JSON.
- Merge new facts.
- Never remove confirmed facts.
- Include:
    symptoms
    duration
    affected_side
    severity
    fever
    cough
    discharge
    vision_changes
    itching
    redness
    medications
    allergies
    chronic_conditions

If something is unknown, don't include it.

Return JSON only.
"""

    try:

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": "You extract structured medical information."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        text = response.output_text.strip()

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1:
            text = text[start:end+1]

        updated = json.loads(text)

        return updated

    except Exception:
        return facts
@app.route("/chat", methods=["POST"])
def chat():

    initialize_session()

    user_message = request.json.get("message", "").strip()

    if not user_message:
        return jsonify({"reply": "Please enter a message."})

    history = session["chat_history"]
    facts = session["known_facts"]

    # Save user message
    history.append({
        "role": "user",
        "content": user_message
    })

    # Update memory using AI
    facts = update_known_facts(history, facts)
    session["known_facts"] = facts

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "system",
            "content": f"KNOWN_FACTS:\n{json.dumps(facts, indent=2)}"
        }
    ]

    messages.extend(history)

    try:

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=messages,
            temperature=0.4,
            max_output_tokens=250
        )

        bot_reply = response.output_text.strip()

        history.append({
            "role": "assistant",
            "content": bot_reply
        })

        # Keep last 20 messages
        session["chat_history"] = history[-20:]

        return jsonify({
            "reply": bot_reply
        })

    except Exception as e:

        return jsonify({
            "reply": str(e)
        })
@app.route("/clear", methods=["POST"])
def clear_chat():

    session.pop("chat_history", None)
    session.pop("known_facts", None)

    return jsonify({
        "status": "success"
    })
@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)