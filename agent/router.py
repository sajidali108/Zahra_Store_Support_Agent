from typing import Literal


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
