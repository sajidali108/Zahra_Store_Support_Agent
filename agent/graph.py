from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from agent.config import get_settings
from agent.prompts import build_system_prompt
from agent.router import (
    IntentCategory,
    classify_intent,
    classify_intent_llm,
    is_off_topic,
)
from agent.tools import get_tools_for_category
from agent.tools import get_zahra_order_detail, track_zahra_order


def _recent_text(history: list[dict[str, Any]] | None, limit: int = 4) -> str:
    items = history or []
    return " ".join(str(item.get("content", "")) for item in items[-limit:])


def _extract_numeric_id(text: str) -> str:
    match = re.search(r"\b\d{3,}\b", text)
    return match.group(0) if match else ""


def _extract_phone_number(text: str) -> str:
    match = re.search(r"\b(\+?92[ -]?3\d{2}[ -]?\d{7}|0?3\d{2}[ -]?\d{7}|\d{10,12})\b", text)
    if match:
        return "".join(c for c in match.group(0) if c.isdigit())
    return ""


def _detect_direct_lookup(
    message: str,
    conversation_history: list[dict[str, Any]] | None,
) -> tuple[str, str, str]:
    numeric_id = _extract_numeric_id(message)
    phone_number = _extract_phone_number(message)
    if not numeric_id or not phone_number or numeric_id == phone_number:
        return "", "", ""

    current = message.lower().strip()
    context = f"{_recent_text(conversation_history)} {message}".lower()
    booking_markers = ("booking", "booking_id", "booking id", "track", "shipment", "parcel", "delivery")
    if any(m in context for m in booking_markers):
        return "booking", numeric_id, phone_number
    return "order", numeric_id, phone_number


def _detect_missing_phone_number(
    message: str,
    conversation_history: list[dict[str, Any]] | None,
) -> str:
    numeric_id = _extract_numeric_id(message)
    if not numeric_id:
        return ""

    context = f"{_recent_text(conversation_history)} {message}".lower()
    phone_number = _extract_phone_number(context)
    if phone_number:
        return ""

    category = classify_intent(message)
    cleaned = message.strip()
    is_just_id = cleaned.isdigit() or (len(cleaned) <= 15 and any(char.isdigit() for char in cleaned))
    
    if category == "order" or is_just_id:
        return numeric_id

    return ""


def _format_direct_result(kind: str, result: dict[str, Any]) -> str:
    if result.get("not_shipped_yet"):
        order_data = result.get("order")
        lines = [
            "**Your order is confirmed but is not yet in the shipping process (it has not been handed over to the courier yet).**\n"
        ]
        if order_data:
            lines.append(f"**Order {order_data.get('order_id', '')}**")
            field_labels = [
                ("customer_name", "Customer"),
                ("customer_email", "Email"),
                ("customer_phone", "Phone"),
                ("shipping_city", "Shipping City"),
                ("payment_status", "Payment Status"),
                ("fulfillment_status", "Delivery Status"),
                ("total_amount", "Total Amount"),
            ]
            for key, label in field_labels:
                if order_data.get(key):
                    lines.append(f"- **{label}:** {order_data[key]}")
            products = order_data.get("products") or []
            if products:
                lines.append("- **Products:**")
                for prod in products:
                    lines.append(f"  - {prod}")
        return "\n".join(lines)

    if result.get("error"):
        return f"**Sorry, I couldn't fetch that {kind}.**\n\n- {result['error']}"

    data = result.get("tracking") if kind == "booking" else result.get("order")
    if not isinstance(data, dict):
        return f"**Sorry, I couldn't fetch that {kind}.**\n\n- Please confirm the id and try again."

    if kind == "booking":
        lines = [f"**Order Tracking {data.get('tracking_number', '')}**".strip()]
        field_labels = [
            ("customer_name", "Customer"),
            ("email", "Email"),
            ("phone", "Phone"),
            ("customer_address", "Shipping Address"),
            ("customer_city", "City"),
            ("courier", "Courier"),
            ("status", "Order Status"),
        ]
        for key, label in field_labels:
            if data.get(key):
                val = str(data[key]).replace("T", " ")
                lines.append(f"- **{label}:** {val}")
                
        history = data.get("history") or []
        if history:
            lines.append("- **Tracking history:**")
            for item in history:
                lines.append(f"  - {item}")
        return "\n".join(lines)

    lines = [f"**Order {data.get('order_id', '')}**".strip()]
    field_labels = [
        ("customer_name", "Customer"),
        ("customer_email", "Email"),
        ("customer_phone", "Phone"),
        ("shipping_city", "Shipping City"),
        ("payment_status", "Payment Status"),
        ("fulfillment_status", "Delivery Status"),
        ("total_amount", "Total Amount"),
    ]
    for key, label in field_labels:
        if data.get(key):
            lines.append(f"- **{label}:** {data[key]}")
    
    products = data.get("products") or []
    if products:
        lines.append("- **Products:**")
        for prod in products:
            lines.append(f"  - {prod}")
    return "\n".join(lines)


async def _direct_lookup_response(
    message: str,
    conversation_history: list[dict[str, Any]] | None,
) -> str:
    kind, lookup_id, phone_number = _detect_direct_lookup(message, conversation_history)
    if not lookup_id or not phone_number:
        return ""
    if kind == "booking":
        result = await track_zahra_order.ainvoke({"order_id": lookup_id, "phone_number": phone_number})
        return _format_direct_result("booking", result)
    if kind == "order":
        result = await get_zahra_order_detail.ainvoke({"order_id": lookup_id, "phone_number": phone_number})
        return _format_direct_result("order", result)
    return ""


class AgentState(TypedDict, total=False):
    message: str
    session_token: str
    conversation_history: list[dict[str, Any]]
    category: IntentCategory
    response: str


def _history_to_messages(
    history: list[dict[str, Any]] | None,
    message: str,
) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for item in history or []:
        role = str(item.get("role", "")).lower()
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        if role in {"assistant", "ai"}:
            messages.append(AIMessage(content=content))
        elif role in {"user", "human"}:
            messages.append(HumanMessage(content=content))

    messages.append(HumanMessage(content=message))
    return messages


async def router_node(state: AgentState) -> AgentState:
    category = await classify_intent_llm(
        state["message"],
        state.get("conversation_history")
    )
    return {"category": category}


def _build_react_agent(
    category: str,
    session_token: str,
    streaming: bool = False,
):
    settings = get_settings()
    tools = get_tools_for_category(category)
    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0,
        streaming=streaming,
    )
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=build_system_prompt(
            session_token=session_token,
            category=category,
        ),
    )


async def agent_node(state: AgentState) -> AgentState:
    if is_off_topic(state["message"]):
        return {
            "response": "I can only help with zahra stores.pk support, like products, orders, shipping, returns, exchanges, and account questions. What store-related question can I help with?"
        }

    conversation_history = state.get("conversation_history")
    message = state["message"]
    direct_response = await _direct_lookup_response(message, conversation_history)
    if direct_response:
        return {"response": direct_response}

    missing_phone_id = _detect_missing_phone_number(message, conversation_history)
    if missing_phone_id:
        return {
            "response": "Thanks! To verify your order, please share the phone number registered with it so I can check its status safely."
        }

    category = state.get("category", "general")
    react_agent = _build_react_agent(
        category=category,
        session_token=state.get("session_token", ""),
        streaming=False,
    )
    result = await react_agent.ainvoke(
        {
            "messages": _history_to_messages(
                conversation_history,
                message,
            )
        }
    )
    final_message = result["messages"][-1]
    return {"response": str(final_message.content)}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("agent", agent_node)
    graph.set_entry_point("router")
    graph.add_edge("router", "agent")
    graph.add_edge("agent", END)
    return graph.compile()


agent_graph = build_graph()


async def run_agent(
    message: str,
    session_token: str,
    conversation_history: list[dict[str, Any]] | None = None,
) -> str:
    result = await agent_graph.ainvoke(
        {
            "message": message,
            "session_token": session_token,
            "conversation_history": conversation_history or [],
        }
    )
    return result["response"]


async def stream_agent(
    message: str,
    session_token: str,
    conversation_history: list[dict[str, Any]] | None = None,
) -> AsyncIterator[str]:
    if is_off_topic(message):
        yield "I can only help with zahra stores.pk support, like products, orders, shipping, returns, exchanges, and account questions. What store-related question can I help with?"
        return

    direct_response = await _direct_lookup_response(message, conversation_history)
    if direct_response:
        yield direct_response
        return

    missing_phone_id = _detect_missing_phone_number(message, conversation_history)
    if missing_phone_id:
        yield "Thanks! To verify your order, please share the phone number registered with it so I can check its status safely."
        return

    model_message = message

    category = await classify_intent_llm(message, conversation_history)
    react_agent = _build_react_agent(
        category=category,
        session_token=session_token,
        streaming=True,
    )

    try:
        async for chunk, _metadata in react_agent.astream(
            {
                "messages": _history_to_messages(
                    conversation_history,
                    model_message,
                )
            },
            stream_mode="messages",
        ):
            if not isinstance(chunk, AIMessageChunk):
                continue
            if isinstance(chunk.content, str) and chunk.content:
                yield chunk.content
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield "\nSorry, I could not complete that response right now. Please try again."
