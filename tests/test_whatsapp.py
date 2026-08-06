import os
import requests
from fastapi import FastAPI, Request, Response
from dotenv import load_dotenv

app = FastAPI()

load_dotenv(override=True)# Read variables from .env
ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN_PER")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")

# Optional: verify they loaded correctly
print("ACCESS_TOKEN:", repr(ACCESS_TOKEN))
print("PHONE_NUMBER_ID:", repr(PHONE_NUMBER_ID))
print("VERIFY_TOKEN:", repr(VERIFY_TOKEN))

GRAPH_URL = f"https://graph.facebook.com/v25.0/{PHONE_NUMBER_ID}/messages"

def send_message(to: str, text: str):
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    print("ACCESS_TOKEN:", repr(ACCESS_TOKEN))
    print("PHONE_NUMBER_ID:", PHONE_NUMBER_ID)

    data = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {
            "preview_url": False,
            "body": text,
        },
    }

    response = requests.post(
        GRAPH_URL,
        headers=headers,
        json=data,
        timeout=10,
    )

    print("Status:", response.status_code)
    print("Response:", response.text)

    return response


def is_valid_whatsapp_message(body: dict) -> bool:
    try:
        value = body["entry"][0]["changes"][0]["value"]

        return (
            "messages" in value
            and len(value["messages"]) > 0
        )

    except Exception:
        return False

def process_whatsapp_message(body: dict):

    value = body["entry"][0]["changes"][0]["value"]

    message = value["messages"][0]

    sender = message["from"]

    message_type = message["type"]

    print("=" * 50)
    print("Sender:", sender)
    print("Type:", message_type)

    if message_type == "text":

        text = message["text"]["body"]

        print("Message:", text)

        reply = f"You said: {text}"

        send_message(sender, reply)

    else:

        print("Unsupported type:", message_type)

 
# ---- Webhook verification (GET) ----
@app.get("/webhook")
async def verify_webhook(request: Request):
    params = request.query_params

    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    print("Mode:", mode)
    print("Received token:", repr(token))
    print("Expected token:", repr(VERIFY_TOKEN))

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return Response(content=challenge, media_type="text/plain")

    return Response(content="Forbidden", status_code=403)

# ---- Receive messages (POST) ----
@app.post("/webhook")
async def receive_webhook(request: Request):

    body = await request.json()

    print(body)

    if is_valid_whatsapp_message(body):

        process_whatsapp_message(body)

    else:

        print("Webhook without a user message")

    return {"status": "ok"}
 
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)