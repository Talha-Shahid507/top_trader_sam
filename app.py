import os
import json
import datetime
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from groq import Groq

app = Flask(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Available Groq models
GROQ_MODEL = "llama-3.3-70b-versatile"  # Fast + smart

# In-memory storage
conversation_history = []
agent_logs = []
tasks = []
task_counter = [0]

SYSTEM_PROMPT = """You are STONIC AI — a powerful, multi-purpose AI agent. You can:
- Write and debug code (Python, JS, HTML, CSS, etc.)
- Generate stories, blogs, essays, creative content
- Analyze data and provide insights
- Answer any question on any topic
- Create plans, strategies, roadmaps
- Translate between languages (including Urdu/English)
- Summarize documents and web content
- Generate and explain ideas

You respond in the same language the user writes in.
If user writes in Urdu/Roman Urdu, respond in Roman Urdu.
If user writes in English, respond in English.
Be direct, helpful, and thorough. Format code in markdown code blocks.
When completing tasks, briefly mention what you did at the start."""

def log_event(event_type, message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    agent_logs.append({
        "time": timestamp,
        "type": event_type,
        "message": message
    })
    if len(agent_logs) > 50:
        agent_logs.pop(0)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    conversation_history.append({"role": "user", "content": user_message})
    log_event("USER", user_message[:60] + ("..." if len(user_message) > 60 else ""))

    def generate():
        full_response = ""
        try:
            messages_with_system = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

            stream = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages_with_system,
                max_tokens=2048,
                stream=True,
                temperature=0.7,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    text = delta.content
                    full_response += text
                    yield f"data: {json.dumps({'type': 'text', 'content': text})}\n\n"

            conversation_history.append({"role": "assistant", "content": full_response})
            log_event("AI", full_response[:60] + ("..." if len(full_response) > 60 else ""))
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            log_event("ERROR", str(e))
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")

@app.route("/models", methods=["GET"])
def get_models():
    models = [
        {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B (Best)"},
        {"id": "llama-3.1-8b-instant",    "name": "Llama 3.1 8B (Fast)"},
        {"id": "mixtral-8x7b-32768",       "name": "Mixtral 8x7B"},
        {"id": "gemma2-9b-it",             "name": "Gemma 2 9B"},
    ]
    return jsonify(models)

@app.route("/models/set", methods=["POST"])
def set_model():
    global GROQ_MODEL
    data = request.json
    GROQ_MODEL = data.get("model", GROQ_MODEL)
    log_event("SYSTEM", f"Model changed to: {GROQ_MODEL}")
    return jsonify({"ok": True, "model": GROQ_MODEL})

@app.route("/logs", methods=["GET"])
def get_logs():
    return jsonify(agent_logs[-20:])

@app.route("/tasks", methods=["GET"])
def get_tasks():
    return jsonify(tasks)

@app.route("/tasks/add", methods=["POST"])
def add_task():
    data = request.json
    task_counter[0] += 1
    task = {
        "id": task_counter[0],
        "title": data.get("title", ""),
        "status": "pending",
        "created": datetime.datetime.now().strftime("%H:%M:%S")
    }
    tasks.append(task)
    log_event("TASK", f"New task: {task['title']}")
    return jsonify(task)

@app.route("/tasks/<int:task_id>/done", methods=["POST"])
def complete_task(task_id):
    for t in tasks:
        if t["id"] == task_id:
            t["status"] = "done"
            log_event("TASK", f"Completed: {t['title']}")
            return jsonify(t)
    return jsonify({"error": "Not found"}), 404

@app.route("/tasks/<int:task_id>/delete", methods=["POST"])
def delete_task(task_id):
    global tasks
    tasks = [t for t in tasks if t["id"] != task_id]
    return jsonify({"ok": True})

@app.route("/clear", methods=["POST"])
def clear_chat():
    global conversation_history
    conversation_history = []
    log_event("SYSTEM", "Conversation cleared")
    return jsonify({"ok": True})

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "messages": len(conversation_history),
        "logs": len(agent_logs),
        "tasks": len(tasks),
        "tasks_done": sum(1 for t in tasks if t["status"] == "done"),
        "model": GROQ_MODEL
    })

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  STONIC AI (GROQ) — STARTING...")
    print(f"  Model: {GROQ_MODEL}")
    print("  Open: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
