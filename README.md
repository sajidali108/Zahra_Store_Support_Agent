# zahra stores.pk Customer Support Agent

FastAPI + LangGraph customer support agent for a Shopify store. The agent routes each message first, then gives the ReAct agent only the tools needed for that intent.

Confirmed Groq reasoning model ID: `llama-3.3-70b-versatile`.

## Setup

```bash
uv sync
```

Fill `.env` with real credentials:

```env
GROQ_API_KEY=your_groq_api_key_here
SHOPIFY_API_KEY=your_shopify_api_key_here
SHOPIFY_ACCESS_TOKEN=your_shopify_access_token_here
SHOPIFY_STORE_URL=https://zahrastores.pk
```

## Build The Policy Knowledge Base

```bash
uv run python -m rag.ingest
```

This scrapes the configured zahra stores.pk policy/contact/service pages, chunks them, embeds them with `sentence-transformers/all-MiniLM-L6-v2`, and stores a local FAISS index at `rag/faiss_index`.

## Run The Backend

```bash
uv run uvicorn main:app --reload
```

The API exposes:

- `GET /health`
- `POST /chat`
- `POST /chat/stream`

Request body:

```json
{
  "message": "Where is my order?",
  "session_token": "customer-session-token",
  "conversation_history": []
}
```

Response body:

```json
{
  "response": "..."
}
```

## Widget

Open `widget/widget.html` or embed its HTML/JS in Shopify. It reads the customer session token silently from:

- `localStorage.shopify_customer_token`
- `shopify_customer_token` cookie
- `_shopify_customer_token` cookie

Then it sends the token with each `/chat` request.

## Notes

Shopify Admin REST does not automatically accept arbitrary browser session tokens as customer identifiers. The scaffolded `get_customer_info` tool supports numeric customer ids and basic customer search strategies, but production should connect the widget token to your store's real customer-auth/session mapping.
