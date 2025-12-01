# app.py (updated)
import os
import json
import time
import logging
from datetime import datetime

import requests
from flask import Flask, request, jsonify

# LangChain imports (same as before)
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# monitoring & retry
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Google Sheets deps
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whatsapp-bot")

app = Flask(__name__)

# ---------------------------
# CONFIG
# ---------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
WHATSAPP_TOKEN = os.environ.get("WHATSAPP_TOKEN")
WHATSAPP_PHONE_ID = os.environ.get("WHATSAPP_PHONE_ID")
VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN", "VERIFY_TOKEN")
GSHEET_CRED_JSON = os.environ.get("GSHEET_CRED_JSON", "/creds/gspread-creds.json")
GSHEET_NAME = os.environ.get("GSHEET_NAME", "WhatsApp Analytics")
VECTOR_STORE_PATH = os.environ.get("VECTOR_STORE_PATH", "/app/data/vector_store")
ORDER_API_URL = os.environ.get("ORDER_API_URL")
ORDER_API_KEY = os.environ.get("ORDER_API_KEY")
PRODUCT_API_URL = os.environ.get("PRODUCT_API_URL")
PRODUCT_API_KEY = os.environ.get("PRODUCT_API_KEY")

# ---------------------------
# Prometheus metrics
# ---------------------------
REQUEST_COUNTER = Counter("whatsapp_requests_total", "Total WhatsApp requests received")
ERROR_COUNTER = Counter("whatsapp_errors_total", "Total errors in processing")

# ---------------------------
# LangChain init
# ---------------------------
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0.2)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def get_retriever():
    try:
        if os.path.exists(VECTOR_STORE_PATH + ".index"):
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings)
            return vectorstore.as_retriever(search_kwargs={"k": 4})
    except Exception as e:
        logger.warning("Could not initialize retriever: %s", e)
    return None

retriever = get_retriever()
if retriever:
    rag_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
else:
    rag_qa = None

# ---------------------------
# External HTTP helpers with retries
# ---------------------------
RETRY_STOP = stop_after_attempt(3)
RETRY_WAIT = wait_exponential(multiplier=1, min=1, max=10)

@retry(stop=RETRY_STOP, wait=RETRY_WAIT, retry=retry_if_exception_type(Exception))
def safe_post(url, headers=None, json_payload=None, timeout=10):
    logger.debug("POST %s", url)
    r = requests.post(url, headers=headers or {}, json=json_payload, timeout=timeout)
    r.raise_for_status()
    return r.json() if r.text else {}

@retry(stop=RETRY_STOP, wait=RETRY_WAIT, retry=retry_if_exception_type(Exception))
def safe_get(url, headers=None, params=None, timeout=10):
    logger.debug("GET %s", url)
    r = requests.get(url, headers=headers or {}, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json() if r.text else {}

# ---------------------------
# Tools: order_lookup & product_finder using real HTTP APIs (sample)
# ---------------------------
def order_lookup_tool(order_id: str) -> str:
    """
    Calls external ORDER_API_URL to get status.
    Set ORDER_API_URL and ORDER_API_KEY in .env
    """
    if not ORDER_API_URL:
        return f"Order {order_id} status: DELIVERED (mock — set ORDER_API_URL to enable real lookup)."
    url = f"{ORDER_API_URL.rstrip('/')}/orders/{order_id}"
    headers = {"Authorization": f"Bearer {ORDER_API_KEY}"} if ORDER_API_KEY else {}
    try:
        resp = safe_get(url, headers=headers)
        # expected response shape: {"order_id": "...", "status": "...", "eta": "..."}
        status = resp.get("status", "unknown")
        eta = resp.get("eta")
        return f"Order {order_id} status: {status}" + (f", ETA: {eta}" if eta else "")
    except Exception as e:
        logger.error("Order lookup failed: %s", e)
        ERROR_COUNTER.inc()
        return f"Order lookup failed for {order_id} (error)."

def product_finder_tool(query: str) -> str:
    """
    Calls PRODUCT_API_URL search endpoint.
    """
    if not PRODUCT_API_URL:
        return f"Found 0 products for '{query}' (mock — set PRODUCT_API_URL to enable real product search)."
    url = f"{PRODUCT_API_URL.rstrip('/')}/search"
    headers = {"Authorization": f"Bearer {PRODUCT_API_KEY}"} if PRODUCT_API_KEY else {}
    try:
        resp = safe_get(url, headers=headers, params={"q": query})
        items = resp.get("items", resp)
        names = [i.get("name") or i.get("title") for i in items[:5]]
        return f"Found {len(names)} products for '{query}': " + ", ".join(names)
    except Exception as e:
        logger.error("Product finder failed: %s", e)
        ERROR_COUNTER.inc()
        return "Product search failed."

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

# Wrap as tools
tools = [
    Tool(name="order_lookup", func=order_lookup_tool, description="Lookup order status by order id."),
    Tool(name="product_finder", func=product_finder_tool, description="Find products by query."),
    Tool(name="faq_lookup", func=faq_tool, description="Answer frequently asked questions from KB.")
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
)

# ---------------------------
# Google Sheets helper
# ---------------------------
def init_gsheets():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(GSHEET_CRED_JSON, scope)
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
        logger.error("Failed to write analytics to Google Sheets: %s", e)
        ERROR_COUNTER.inc()

# ---------------------------
# WhatsApp helper
# ---------------------------
@retry(stop=RETRY_STOP, wait=RETRY_WAIT, retry=retry_if_exception_type(Exception))
def send_whatsapp_message(to: str, message: str):
    url = f"https://graph.facebook.com/v17.0/{WHATSAPP_PHONE_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": message}}
    r = requests.post(url, headers=headers, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()

# ---------------------------
# Webhook endpoints
# ---------------------------
@app.route("/metrics")
def metrics():
    resp = generate_latest()
    return (resp, 200, {"Content-Type": CONTENT_TYPE_LATEST})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat()}), 200

@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    REQUEST_COUNTER.inc()
    if request.method == "GET":
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")
        if token == VERIFY_TOKEN:
            return challenge, 200
        return "Invalid verification token", 403

    data = request.json
    logger.info("Incoming webhook (truncated): %s", json.dumps(data)[:800])
    try:
        msg = data["entry"][0]["changes"][0]["value"]["messages"][0]
        sender = msg["from"]
        text = msg.get("text", {}).get("body", "")
        timestamp = int(msg.get("timestamp", time.time()))
    except Exception as e:
        logger.error("Malformed webhook payload: %s", e)
        return jsonify({"status": "ignored"}), 200

    answer = None
    if retriever:
        try:
            rag_answer = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever).run(text)
            if rag_answer and len(rag_answer.strip()) > 20:
                answer = rag_answer
                logger.info("RAG answered.")
        except Exception as e:
            logger.warning("RAG error: %s", e)

    if not answer:
        try:
            prompt = f"You are a friendly support assistant. User message: {text}"
            answer = agent.run(prompt)
        except Exception as e:
            logger.error("Agent error: %s", e)
            ERROR_COUNTER.inc()
            chain = ConversationChain(llm=llm, memory=memory, verbose=False)
            answer = chain.predict(input=text)

    try:
        send_whatsapp_message(sender, answer)
    except Exception as e:
        logger.error("Failed to send WhatsApp msg: %s", e)
        ERROR_COUNTER.inc()

    # log analytics
    try:
        log_analytics([datetime.utcfromtimestamp(timestamp).isoformat(), sender, text, answer])
    except Exception as e:
        logger.error("Analytics logging failed: %s", e)
        ERROR_COUNTER.inc()

    return jsonify({"status": "ok"}), 200

# lightweight order lookup API for direct calls
@app.route("/order/<order_id>", methods=["GET"])
def order_lookup(order_id):
    try:
        result = order_lookup_tool(order_id)
        return jsonify({"order_id": order_id, "result": result}), 200
    except Exception as e:
        logger.error("Order lookup endpoint failed: %s", e)
        ERROR_COUNTER.inc()
        return jsonify({"error": "lookup failed"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
