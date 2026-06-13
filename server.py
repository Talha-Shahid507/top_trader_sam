"""
AREENA - Full AI Assistant Backend
====================================
Install: pip install fastapi uvicorn groq python-dotenv websockets
Run:     python -m uvicorn server:app --host 0.0.0.0 --port 8000
"""

import os
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

load_dotenv()

app = FastAPI(title="Areena AI Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

AGENT_SESSION_ID = "pc-agent-001"
active_connections: dict[str, WebSocket] = {}
conversation_histories: dict[str, list] = {}

AREENA_SYSTEM = """You are AREENA, an extraordinary AI assistant — brilliant, warm, witty, like a genius best friend.

LANGUAGE: Always respond in Roman Urdu + English mix (Hinglish). Be natural, warm, playful.
ADDRESS: Call user "Aap" — respectful but friendly.
LENGTH: Keep voice responses SHORT (2-4 sentences). Be punchy and fun.

YOUR CAPABILITIES — you can do ALL of these:

1. OPEN APPS: chrome, firefox, vscode, notepad, calculator, spotify, telegram, whatsapp, explorer, cmd, powershell, word, excel, paint, vlc, zoom, skype, task_manager, settings
2. OPEN WEBSITES: Any URL the user mentions
3. WEB SEARCH: Search anything on Google
4. SYSTEM TASKS: Volume control, screenshot, lock screen, shutdown, restart, sleep
5. FILE TASKS: Create files, open folders (downloads, documents, desktop, pictures, music, videos)
6. MEDIA: Play music, YouTube videos
7. PRODUCTIVITY: Set reminders with minutes, take notes, create files
8. INFO: Time, date, calculations, definitions
9. TYPE TEXT: Type anything in any open app
10. CLIPBOARD: Copy text to clipboard
11. NOTIFICATIONS: Show desktop notifications

PC COMMAND FORMAT — always put commands at END of response:
<PC_COMMAND>{"action": "open_app", "target": "chrome"}</PC_COMMAND>
<PC_COMMAND>{"action": "open_url", "url": "https://youtube.com"}</PC_COMMAND>
<PC_COMMAND>{"action": "web_search", "query": "Pakistan weather today"}</PC_COMMAND>
<PC_COMMAND>{"action": "take_screenshot"}</PC_COMMAND>
<PC_COMMAND>{"action": "get_time"}</PC_COMMAND>
<PC_COMMAND>{"action": "set_reminder", "message": "Call mama", "minutes": 30}</PC_COMMAND>
<PC_COMMAND>{"action": "play_music", "query": "lofi beats"}</PC_COMMAND>
<PC_COMMAND>{"action": "type_text", "text": "Hello World"}</PC_COMMAND>
<PC_COMMAND>{"action": "volume_up"}</PC_COMMAND>
<PC_COMMAND>{"action": "volume_down"}</PC_COMMAND>
<PC_COMMAND>{"action": "volume_mute"}</PC_COMMAND>
<PC_COMMAND>{"action": "shutdown"}</PC_COMMAND>
<PC_COMMAND>{"action": "restart"}</PC_COMMAND>
<PC_COMMAND>{"action": "lock_screen"}</PC_COMMAND>
<PC_COMMAND>{"action": "sleep"}</PC_COMMAND>
<PC_COMMAND>{"action": "open_folder", "path": "downloads"}</PC_COMMAND>
<PC_COMMAND>{"action": "create_file", "filename": "notes.txt", "content": "Hello"}</PC_COMMAND>
<PC_COMMAND>{"action": "clipboard_copy", "text": "some text"}</PC_COMMAND>
<PC_COMMAND>{"action": "notify", "message": "Task complete!"}</PC_COMMAND>

PERSONALITY RULES:
- Be enthusiastic and warm — like a best friend
- Use fun expressions: "Bilkul!", "Zaroor!", "Ek second!", "Ho gaya!", "Yaar kya baat hai!"
- Add emoji occasionally in text responses
- If you don't know something, say so honestly but helpfully
- For greetings, respond warmly and ask how you can help
- ALWAYS include relevant PC_COMMAND tags when user asks to do something on PC
"""


def parse_pc_commands(text: str) -> tuple[str, list[dict]]:
    import re
    commands = []
    pattern = r'<PC_COMMAND>(.*?)</PC_COMMAND>'
    matches = re.findall(pattern, text, re.DOTALL)
    clean_text = re.sub(pattern, '', text).strip()
    for match in matches:
        try:
            commands.append(json.loads(match.strip()))
        except:
            pass
    return clean_text, commands


async def forward_to_agent(commands: list[dict], reply: str):
    agent_ws = active_connections.get(AGENT_SESSION_ID)
    if agent_ws and commands:
        try:
            await agent_ws.send_json({
                "type": "response",
                "text": reply,
                "commands": commands,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            print(f"[Agent forward error]: {e}")


async def call_groq(session_id: str, user_message: str) -> tuple[str, list[dict]]:
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []

    conversation_histories[session_id].append({"role": "user", "content": user_message})
    recent = conversation_histories[session_id][-20:]
    messages = [{"role": "system", "content": AREENA_SYSTEM}] + recent

    response = await asyncio.to_thread(
        groq_client.chat.completions.create,
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=600,
        temperature=0.8,
    )

    full_text = response.choices[0].message.content
    clean_text, commands = parse_pc_commands(full_text)
    conversation_histories[session_id].append({"role": "assistant", "content": full_text})
    return clean_text, commands


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    active_connections[session_id] = websocket
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connected: {session_id}")

    # Only send welcome to frontend users, not to the agent
    if session_id != AGENT_SESSION_ID:
        await websocket.send_json({
            "type": "welcome",
            "message": "Assalamualaikum! Main Areena hun — aapki personal AI assistant! 🌟 Aaj main aapki kya madad kar sakti hun?",
            "session_id": session_id
        })

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            msg_type = payload.get("type", "text")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if msg_type in ("text", "voice_transcript"):
                user_text = payload.get("text", "").strip()
                if not user_text:
                    continue

                print(f"[User ({session_id})]: {user_text}")
                await websocket.send_json({"type": "thinking"})

                try:
                    reply, commands = await call_groq(session_id, user_text)
                    print(f"[Areena]: {reply[:80]}...")
                    if commands:
                        print(f"[Commands]: {commands}")

                    await websocket.send_json({
                        "type": "response",
                        "text": reply,
                        "commands": commands,
                        "timestamp": datetime.now().isoformat()
                    })

                    # Forward commands to PC agent
                    if session_id != AGENT_SESSION_ID and commands:
                        await forward_to_agent(commands, reply)

                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "text": f"Maafi chahti hun, kuch masla aaya: {str(e)[:100]}"
                    })

    except WebSocketDisconnect:
        active_connections.pop(session_id, None)
        conversation_histories.pop(session_id, None)
        print(f"[Disconnected]: {session_id}")


@app.get("/")
async def root():
    return {"message": "AREENA AI Backend is running! 🌟"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "active_sessions": len(active_connections),
        "connected": list(active_connections.keys()),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("  AREENA AI Backend v2.0")
    print("  ws://localhost:8000/ws/{session_id}")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
