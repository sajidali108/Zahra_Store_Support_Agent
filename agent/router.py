from typing import Any, Literal
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from agent.config import get_settings

IntentCategory = Literal["order", "product", "policy", "account", "general"]


ORDER_KEYWORDS = {
    "order",
    "orders",
    "tracking",
    "track",
    "delivery",
    "delivered",
    "shipment",
    "shipping status",
    "parcel",
    "package",
    "history",
    "invoice",
    "cancel",
}

PRODUCT_KEYWORDS = {
    "product",
    "products",
    "price",
    "stock",
    "available",
    "availability",
    "category",
    "categories",
    "size",
    "color",
    "variant",
    "collection",
    "catalog",
}

POLICY_KEYWORDS = {
    "policy",
    "policies",
    "return",
    "refund",
    "exchange",
    "privacy",
    "terms",
    "condition",
    "conditions",
    "shipping policy",
    "contact",
    "service",
    "services",
}

ACCOUNT_KEYWORDS = {
    "account",
    "profile",
    "address",
    "phone",
    "email",
    "customer",
    "login",
    "password",
}

OFF_TOPIC_KEYWORDS = {
    "homework",
    "assignment",
    "essay",
    "math problem",
    "solve this",
    "write code",
    "programming",
    "coding",
    "recipe",
    "medical advice",
    "legal advice",
}

def is_off_topic(message: str) -> bool:
    text = f" {message.lower()} "
    return any(keyword in text for keyword in OFF_TOPIC_KEYWORDS)


def classify_intent(message: str) -> IntentCategory:
    text = f" {message.lower()} "

    def has_any(keywords: set[str]) -> bool:
        return any(keyword in text for keyword in keywords)

    if has_any(ORDER_KEYWORDS):
        return "order"
    if has_any(PRODUCT_KEYWORDS):
        return "product"
    if has_any(POLICY_KEYWORDS):
        return "policy"
    if has_any(ACCOUNT_KEYWORDS):
        return "account"
    return "general"


async def classify_intent_llm(
    message: str,
    conversation_history: list[dict[str, Any]] | None = None,
) -> IntentCategory:
    # 1. Format the history context (up to the last 4 turns)
    history_items = conversation_history or []
    recent = history_items[-4:]
    history_context = " ".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in recent)

    try:
        settings = get_settings()
        llm = ChatGroq(
            api_key=settings.groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=5,
        )
        system_prompt = (
            "You are an intent classifier for a customer support chatbot.\n"
            "Given the user's message and recent context, classify the active intent into exactly one of these categories:\n"
            "- \"order\": for order status, tracking, cancellations, order details lookup.\n"
            "- \"product\": for product search, availability, catalog details.\n"
            "- \"policy\": for returns, exchanges, refunds, shipping times, general FAQs.\n"
            "- \"account\": for customer login, password, registration inquiries.\n"
            "- \"general\": for greetings, general chit-chat, off-topic, or fallback.\n\n"
            "Output ONLY the category name in lowercase. Do not write any explanations or other words."
        )
        user_content = f"Recent Context: {history_context}\nUser Message: {message}"
        
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ])
        category = str(response.content).strip().lower().strip(" \t\n\r.\"'`(),[]{}")
        if category in {"order", "product", "policy", "account", "general"}:
            return category
    except Exception:
        # Fallback will run below
        pass

    # Fallback to keyword-based classification on context
    context = f"{history_context} {message}"
    return classify_intent(context)
