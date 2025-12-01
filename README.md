# LangChain WhatsApp Bot

This repository contains a LangChain-based WhatsApp chatbot using RAG (Retrieval Augmented Generation) and custom tools for Order Lookup, Product Finder, and FAQ responses.

## Features

* WhatsApp Cloud webhook integration
* LangChain agent with custom tools
* RAG bot using OpenAI embeddings and FAISS
* Google Sheets analytics logging
* Dockerized with docker-compose for easy deployment
* Prometheus metrics for monitoring

## 1. Setup

1. Clone this repository
2. Create `.env` file with the following variables:

```
OPENAI_API_KEY=sk-...
WHATSAPP_TOKEN=EAAX...
WHATSAPP_PHONE_ID=1234567890
WHATSAPP_VERIFY_TOKEN=VERIFY_TOKEN
GSHEET_CRED_JSON=/creds/gspread-creds.json
GSHEET_NAME=WhatsApp Analytics
VECTOR_STORE_PATH=/app/data/vector_store
ORDER_API_URL=https://api.example.com/orders
ORDER_API_KEY=xxxx
PRODUCT_API_URL=https://api.example.com/products
PRODUCT_API_KEY=yyyy
```

3. Place your Google service account JSON at `./creds/gspread-creds.json`
4. Put documents to index for RAG into `./docs_to_index/`
5. Build vectorstore locally:

```
python rag_setup.py
```

## 2. Build & Run

```
docker-compose up --build -d
```

Webhook endpoint: `https://your-domain/webhook`

## 3. n8n Workflow

Import `whatsapp_to_langchain_workflow.json` into n8n. Configure the HTTP Request nodes and Google Sheets node with correct credentials.

---

## 6. Deployment Guide

### n8n Cloud / Self-Hosted

1. Import the workflow JSON
2. Add API credentials for OpenAI, WhatsApp, Google Sheets, etc.
3. Replace placeholder variables:

```
{{YOUR_API_KEY}}
{{YOUR_SENDER_EMAIL}}
{{YOUR_WHATSAPP_TOKEN}}
```

4. Enable retries and set timeouts as needed

### Environment Variables

```
N8N_ENCRYPTION_KEY=xxxxx
OPENAI_API_KEY=xxxxx
WHATSAPP_TOKEN=xxxxx
DB_URL=xxxxx
```

Restart n8n after editing the `.env` file.

## 7. Scaling & Optimization Tips

* Split large workflows into sub-workflows
* Use SplitInBatches for large CSVs or API payloads
* Use expressions instead of multiple Set nodes
* Use Redis/Postgres for frequent executions
* For AI workflows: enforce output formatting, token limits, and system prompts

## 8. Security & Compliance Notes

### Data Security

* Store credentials using n8n encryption
* Do not commit API keys
* Restrict external HTTP nodes by domain
* Secure webhooks with signature verification or IP allowlists

### Compliance

* GDPR, NDPR, HIPAA as applicable
* Mask personal data before sending to LLMs

## 9. Troubleshooting & Common Errors

### 401 Unauthorized / Invalid Credentials

* Recreate credentials, ensure token validity, reconnect OAuth accounts

### Workflow stops after Webhook node

* Ensure workflow is activated, test with URL preview

### Too many requests (429)

* Add Wait node, enable retry logic, reduce batch sizes

### AI output malformed JSON

* Enforce JSON in system prompt, validate with IF node, optionally fix using Function node

### WhatsApp Cloud API — Message Not Delivered

* Verify phone number, opt-in, approved template, correct endpoint

### RAG bot not retrieving correct document

* Ensure embeddings match model, adjust similarity threshold, re-ingest documents

### Google Sheets: Quota Exceeded

* Add queueing, cache data, consider PostgreSQL for high volume

### CRM Sync duplication

* Use unique identifiers, add Merge node and IF node to prevent duplicates
