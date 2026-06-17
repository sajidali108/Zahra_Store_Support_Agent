# Zahra Stores customer support agent

FastAPI + LangGraph customer support agent for zahra stores.pk. The agent utilizes intent routing to classify user messages, directing them to specialized ReAct agent executors with tool subsets or triggering direct programmatic bypasses for fast response latency and security validation.

Confirmed Groq reasoning model ID: `openai/gpt-oss-120b`.

---

## Key Features

1. **Structured Intent Routing**: First classifies customer messages into categories (`order`, `product`, `policy`, `account`, `general`) and passes the message to a sub-agent equipped with only the necessary tools.
2. **Order Tracking Security Intercept**: 
   * **Direct Bypass**: If the customer provides both their numeric order ID and registered phone number in a message, it completely bypasses the LLM and queries the APIs directly for instant response delivery.
   * **Verification Intercept**: Programmatically requests the registered phone number immediately when an order ID is provided alone, bypassing LLM reasoning loops.
3. **Refined Tracking Timeline**:
   * Renames label to `"Order Status"`, maps completed/delivered states to `"Delivered"`, and hides internal technical courier statuses (like `DeManifested` and `Delivered` checkpoints).
   * Collapses multi-phase courier delivery milestones into a single `"Parcel is out for delivery."` event.
   * Embeds ordered/placed dates chronologically directly within the timeline.
   * Masks only the trailing 30% of the customer address to balance privacy and utility.
4. **Policy Search (RAG)**: Integrates vector-based retrieval over scraped policy, shipping, return, exchange, and contact FAQ pages.

---

## Setup & Installation

### 1. Synchronize Dependencies
Make sure you have `uv` installed, then synchronize packages:
```bash
uv sync
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory and fill it with your credentials:
```env
# Groq LLM configuration
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b

# Zahra Courier API configuration
ZAHRA_API_BASE_URL=https://api.example.com
ZAHRA_API_TOKEN=your_zahra_secret_token_here

# Shopify Admin REST configuration
SHOPIFY_STORE_URL=https://zahrastores.pk
SHOPIFY_API_VERSION=2024-04
SHOPIFY_ACCESS_TOKEN=your_shopify_access_token_here
```

---

## Policy Knowledge Base Scraper (RAG)

Before querying policies, build the local FAISS vector search database:
```bash
uv run python -m rag.ingest
```
This scrapes the configured policy/contact pages, chunks the content, embeds them using `sentence-transformers/all-MiniLM-L6-v2`, and generates a local index at `rag/faiss_index`.

---

## Run The Server

Start the hot-reloading FastAPI application:
```bash
uv run uvicorn main:app --reload
```

---

## API Documentation

### POST `/chat/stream` (Streaming Response)
Request body:
```json
{
  "message": "Track my order #183024",
  "session_token": "customer-session-token",
  "conversation_history": [
    {"role": "user", "content": "Hi!"},
    {"role": "assistant", "content": "Hello! How can I help you today?"}
  ]
}
```

### POST `/chat` (Non-streaming Response)
Request body is identical to `/chat/stream`. Returns:
```json
{
  "response": "To verify order #183024, please share the phone number registered with it so I can check its status safely."
}
```

---

## Widget Deployment

The chat widget files are located in `/widget`. To deploy, you can open `widget/widget.html` or embed its HTML/JS/CSS assets into Shopify. It automatically reads the customer session token silently from local storage and cookies (`shopify_customer_token` or `_shopify_customer_token`) and appends it to `/chat` requests.
