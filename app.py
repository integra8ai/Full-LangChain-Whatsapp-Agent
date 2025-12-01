# app.py
import os
import json
from flask import Flask, request, jsonify
import requests
from datetime import datetime
import logging

# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.tools import Tool

# Google Sheets
import gspread
from oauth2client.service_account import ServiceAccountCredentials

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# ---------------------------
# CONFIG - set these env vars
# ---------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
WHATSAPP_TOKEN = os.environ.get("WHATSAPP_TOKEN")
WHATSAPP_PHONE_ID = os.environ.get("WHATSAPP_PHONE_ID")
VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN", "VERIFY_TOKEN")
GSHEET_CRED_JSON = os.environ.get("GSHEET_CRED_JSON", "/creds/gspread-creds.json")
GSHEET_NAME = os.environ.get("GSHEET_NAME", "WhatsApp Analytics")

VECTOR_STORE_PATH = os.environ.get("VECTOR_STORE_PATH", "vector_store.faiss")

# ---------------------------
# Init LLM, memory, tools
# ---------------------------
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0.2)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# RAG retriever if vector DB exists
def get_retriever():
    if os.path.exists(VECTOR_STORE_PATH + ".index"):
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 4})
    return None

retriever = get_retriever()
if retriever:
    rag_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
else:
    rag_qa = None

# Tool: Order lookup (mock; replace with real API)
def order_lookup_tool(order_id: str) -> str:
    # TODO: replace with real order service API call
    return f"Order {order_id} status: DELIVERED (mock)."

# Tool: Product finder (mock)
def product_finder_tool(query: str) -> str:
    # TODO: replace with product DB/API call
    return f"Found 3 products for '{query}': Product A, Product B, Product C (mock)."

# Tool: FAQ lookup (simple keyword match)
FAQ_KB = {
    "refund": "Our refund policy: 30 days money-back.",
    "shipping": "We ship in 3-5 business days.",
    "pricing": "Please visit our pricing page: https://example.com/pricing"
}

def faq_tool(query: str) -> str:
    q = query.lower()
    for k, v in FAQ_KB.items():
        if k in q:
            return v
    return "I couldn't find an FAQ answer — I can connect you to support."

# Wrap as LangChain Tools
tools = [
    Tool(name="order_lookup", func=order_lookup_tool, description="Lookup order status by order id."),
    Tool(name="product_finder", func=product_finder_tool, description="Find products by query."),
    Tool(name="faq_lookup", func=faq_tool, description="Answer frequently asked questions from KB.")
]

# Initialize an agent that can use the tools + RAG if available
agent = initialize_agent(
    tools + ([] if not retriever else []),
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
)

# If we want to combine RAG for retrieval then follow up with agent, we'll call those separately.

# ---------------------------
# Google Sheets helper
# ---------------------------
def init_gsheets():
    scope = ["https://spreadsheets.google.com/feeds",
             "https://www.googleapis.com/auth/drive"]
    creds_json = GSHEET_CRED_JSON
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_json, scope)
    client = gspread.authorize(creds)
    try:
        sheet = client.open(GSHEET_NAME).sheet1
    except Exception:
        sheet = client.create(GSHEET_NAME).sheet1
    return sheet

def log_analytics(row: list):
    try:
        sheet = init_gsheets()
        sheet.append_row(row)
    except Exception as e:
        app.logger.error("Failed to write analytics to Google Sheets: %s", e)

# ---------------------------
# WhatsApp helper
# ---------------------------
def send_whatsapp_message(to: str, message: str):
    url = f"https://graph.facebook.com/v17.0/{WHATSAPP_PHONE_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message}
    }
    r = requests.post(url, headers=headers, json=payload)
    app.logger.info("WhatsApp send response: %s", r.text)
    return r

# ---------------------------
# Webhook -- verify & receive
# ---------------------------
@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    # verification (GET)
    if request.method == "GET":
        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")
        if token == VERIFY_TOKEN:
            return challenge, 200
        return "Invalid verification token", 403

    data = request.json
    app.logger.info("Incoming webhook: %s", json.dumps(data)[:1000])
    # extract message
    try:
        msg = data["entry"][0]["changes"][0]["value"]["messages"][0]
        sender = msg["from"]
        text = msg.get("text", {}).get("body", "")
        timestamp = int(msg.get("timestamp", datetime.utcnow().timestamp()))
    except Exception as e:
        app.logger.error("Malformed webhook payload: %s", e)
        return jsonify({"status": "ignored"}), 200

    # 1) RAG priority: if we have a vector DB, try to answer from it
    answer = None
    if rag_qa:
        try:
            rag_answer = rag_qa.run(text)
            if rag_answer and len(rag_answer.strip()) > 20:
                answer = rag_answer
                app.logger.info("RAG answered.")
        except Exception as e:
            app.logger.error("RAG error: %s", e)

    # 2) Else let the agent (which can call tools) try to answer
    if not answer:
        try:
            # agent.run expects a string; it can call our tools
            # We provide a short prefix to encourage helpfulness
            prompt = f"You are a friendly support assistant. User message: {text}"
            answer = agent.run(prompt)
        except Exception as e:
            app.logger.error("Agent error: %s", e)
            # fallback to simple chain (conversation)
            chain = ConversationChain(llm=llm, memory=memory, verbose=False)
            answer = chain.predict(input=text)

    # send reply
    try:
        send_whatsapp_message(sender, answer)
    except Exception as e:
        app.logger.error("Failed to send WhatsApp msg: %s", e)

    # log analytics (timestamp, sender, text, answer)
    try:
        log_analytics([datetime.utcfromtimestamp(timestamp).isoformat(), sender, text, answer])
    except Exception as e:
        app.logger.error("Analytics logging failed: %s", e)

    return jsonify({"status": "ok"}), 200

# ---------------------------
# Analytics endpoint (n8n can post here)
# ---------------------------
@app.route("/analytics", methods=["POST"])
def analytics():
    payload = request.json
    # accept expected fields: sender, message, intent, sentiment, extra
    row = [
        payload.get("timestamp", datetime.utcnow().isoformat()),
        payload.get("sender"),
        payload.get("message"),
        payload.get("intent"),
        payload.get("sentiment"),
        payload.get("notes")
    ]
    log_analytics(row)
    return jsonify({"status": "logged"}), 200

# ---------------------------
# Order lookup endpoint (optional direct API)
# ---------------------------
@app.route("/order/<order_id>", methods=["GET"])
def order_lookup(order_id):
    # Production: replace with real API request to your order system
    result = order_lookup_tool(order_id)
    return jsonify({"order_id": order_id, "result": result})

# ---------------------------
# Health
# ---------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat()}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
