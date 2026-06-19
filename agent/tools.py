from __future__ import annotations

import asyncio
import re
from typing import Any

import httpx
from langchain_core.tools import tool

from agent.config import get_settings
from rag.retriever import search_knowledge_base as rag_search_knowledge_base


def _shopify_base_url() -> str:
    settings = get_settings()
    store_url = settings.shopify_store_url.rstrip("/")
    return f"{store_url}/admin/api/{settings.shopify_api_version}"


def _shopify_headers() -> dict[str, str]:
    settings = get_settings()
    return {
        "X-Shopify-Access-Token": settings.shopify_access_token,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


async def _shopify_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{_shopify_base_url()}/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(url, headers=_shopify_headers(), params=params)
        response.raise_for_status()
        return response.json()


def _zahra_base_url() -> str:
    return get_settings().zahra_api_base_url.rstrip("/")


def _zahra_headers() -> dict[str, str]:
    settings = get_settings()
    return {
        "X-SECRET-KEY": settings.zahra_api_token.strip(),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


async def _zahra_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{_zahra_base_url()}/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(url, headers=_zahra_headers(), params=params)
        response.raise_for_status()
        return response.json()


async def _zahra_post(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{_zahra_base_url()}/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(url, headers=_zahra_headers(), json=payload or {})
        response.raise_for_status()
        return response.json()


async def _zahra_post_form(path: str, data: dict[str, Any]) -> dict[str, Any]:
    url = f"{_zahra_base_url()}/{path.lstrip('/')}"
    headers = _zahra_headers()
    headers.pop("Content-Type", None)
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(url, headers=headers, data=data)
        response.raise_for_status()
        return response.json()


def _summarize_order(order: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": order.get("id"),
        "name": order.get("name"),
        "order_number": order.get("order_number"),
        "created_at": order.get("created_at"),
        "financial_status": order.get("financial_status"),
        "fulfillment_status": order.get("fulfillment_status") or "unfulfilled",
        "total_price": order.get("total_price"),
        "currency": order.get("currency"),
        "tracking_numbers": [
            number
            for fulfillment in order.get("fulfillments", [])
            for number in fulfillment.get("tracking_numbers", [])
        ],
        "tracking_urls": [
            url
            for fulfillment in order.get("fulfillments", [])
            for url in fulfillment.get("tracking_urls", [])
        ],
    }


def _find_first_value(data: Any, keys: set[str]) -> Any:
    if isinstance(data, dict):
        for key, value in data.items():
            normalized = key.lower().replace(" ", "_").replace("-", "_")
            if normalized in keys and value not in (None, "", []):
                return value
        for value in data.values():
            found = _find_first_value(value, keys)
            if found not in (None, "", []):
                return found
    elif isinstance(data, list):
        for item in data:
            found = _find_first_value(item, keys)
            if found not in (None, "", []):
                return found
    return None


def _collect_product_names(data: Any) -> list[str]:
    names: list[str] = []
    product_name_keys = {"product_name", "product_title", "title", "name", "item_name"}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                normalized = key.lower().replace(" ", "_").replace("-", "_")
                if normalized in product_name_keys and isinstance(item, str) and item.strip():
                    names.append(item.strip())
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(data)
    deduped: list[str] = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped[:5]


def _normalize_phone(phone: str | None) -> str:
    if not phone:
        return ""
    digits = "".join(char for char in phone if char.isdigit())
    return digits[-10:] if len(digits) >= 10 else digits


def _mask_address(address: str | None) -> str:
    if not address:
        return ""
    address = address.strip()
    addr_len = len(address)
    if addr_len <= 5:
        return address
    mask_count = int(addr_len * 0.3)
    if mask_count < 1:
        mask_count = 1
    keep_len = addr_len - mask_count
    return address[:keep_len] + "*" * mask_count


def _format_datetime(dt_str: str | None) -> str:
    if not dt_str:
        return ""
    dt_str = dt_str.replace("T", " ")
    if "." in dt_str:
        dt_str = dt_str.split(".")[0]
    return dt_str.strip()


def _elaborate_status(status: str, reason: str | None = None) -> str:
    status_lower = status.strip().lower()
    mappings = {
        "booked": "Order booked with the courier and package is being prepared.",
        "shipment received": "Shipment received at the courier origin facility.",
        "shipment arrived": "Shipment arrived at the courier transit hub.",
        "forward": "Parcel forwarded to the destination city delivery center.",
        "demanifested": "Shipment unloaded and being sorted at the local delivery center.",
        "dlvry phase i": "Parcel is out for delivery (Phase I). The rider is on the way.",
        "dlvry phase ii": "Parcel is out for delivery (Phase II). Please ensure your phone is reachable.",
        "delivered": "Successfully delivered to the customer.",
        "undelivered": "Delivery attempt failed.",
        "returned": "Shipment returned to sender.",
        "returned to origin": "Shipment returned to the origin facility.",
        "refused": "Delivery refused by the customer.",
        "canceled": "Booking cancelled.",
    }
    
    desc = mappings.get(status_lower)
    if not desc:
        if "book" in status_lower:
            desc = "Order booked with the courier."
        elif "receiv" in status_lower:
            desc = "Shipment received by the courier."
        elif "arriv" in status_lower:
            desc = "Shipment arrived at the courier hub."
        elif "dlvry" in status_lower or "deliver" in status_lower:
            if "out" in status_lower or "phase" in status_lower:
                desc = "Parcel is out for delivery. The rider is en route."
            else:
                desc = "Successfully delivered."
        elif "return" in status_lower:
            desc = "Parcel is being returned to our warehouse."
        else:
            desc = f"Status update: {status}."

    if reason and reason.strip():
        desc += f" Reason: {reason.strip()}"
    return desc


def _summarize_zahra_order_detail(response: dict[str, Any], order_id: str) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {"error": "Invalid response format"}
    
    # Support both {"data": {...}} wrapper and direct {...} order dict
    data = response.get("data") if isinstance(response.get("data"), dict) else response
    
    shipping = data.get("shipping_address") or {}
    customer = data.get("customer") or {}
    line_items = data.get("line_items") or []

    products = []
    for item in line_items:
        title = item.get("title")
        qty = item.get("quantity")
        if title:
            products.append(f"{title} (Qty: {qty})" if qty else title)

    # Extract payment/fulfillment details
    payment = data.get("financial_status")
    fulfillment = data.get("fulfillment_status")
    if not fulfillment:
        fulfillment = "Unfulfilled (Not yet shipped)"

    return {
        "order_id": data.get("id") or order_id,
        "order_number": data.get("order_number") or data.get("name") or order_id,
        "created_at": data.get("created_at"),
        "payment_status": payment,
        "fulfillment_status": fulfillment,
        "total_amount": f"{data.get('total_price')} {data.get('currency', 'PKR')}",
        "customer_name": shipping.get("name") or customer.get("name") or (f"{shipping.get('first_name', '')} {shipping.get('last_name', '')}".strip()),
        "customer_email": customer.get("email"),
        "customer_phone": shipping.get("phone"),
        "shipping_city": shipping.get("city"),
        "products": products,
    }


def _summarize_zahra_tracking(
    response: dict[str, Any], 
    tracking_number: str,
    ordered: str | None = None,
) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {"error": "Invalid response format"}
    
    # Support both {"data": {...}} wrapper and direct {...} dict
    data = response.get("data") if isinstance(response.get("data"), dict) else response
    
    history = data.get("history") or []
    if not isinstance(history, list):
        history = []
        
    # Chronologically sort history by datetime to ensure index -1 is always the latest
    def get_dt(item):
        return (item.get("datetime") or "").replace("T", " ")
        
    try:
        sorted_history = sorted(history, key=get_dt)
    except Exception:
        sorted_history = history

    # Determine current status
    api_delivery_status = (data.get("delivery_status") or "").strip().lower()
    api_status = (data.get("status") or "").strip().lower()
    
    # We check history checkpoints for out of delivery status
    latest_event_raw = sorted_history[-1] if sorted_history else {}
    latest_raw_status = (latest_event_raw.get("status") or "").strip().lower()

    if api_delivery_status == "delivered" or api_status == "completed":
        latest_status = "Delivered"
    elif "dlvry phase" in latest_raw_status:
        latest_status = "Out for delivery"
    else:
        # Filter out demanifested and dlvry phase checkpoints to find the last general checkpoint
        general_history = [
            item for item in sorted_history 
            if (item.get("status") or "").strip().lower() not in {"demanifested", "dlvry phase i", "dlvry phase ii"}
        ]
        latest_event = general_history[-1] if general_history else {}
        status_val = latest_event.get("status")
        if not status_val:
            # Fallback to the top-level API status or delivery status
            status_val = data.get("status") or data.get("delivery_status")
            
        if status_val:
            status_lower = str(status_val).strip().lower()
            if status_lower == "booked":
                latest_status = "Booked with courier"
            else:
                latest_status = str(status_val).strip().title()
        else:
            latest_status = "Unknown"

    # Construct the timeline history
    formatted_history = []
    if ordered:
        formatted_history.append(f"Order Placed at {_format_datetime(ordered)}")
    
    booking_date_val = data.get("booking_at") or data.get("booking_date")
    if booking_date_val:
        formatted_history.append(f"Order Booked at {_format_datetime(booking_date_val)}")

    has_out_for_delivery = False
    for item in sorted_history:
        status_str = (item.get("status") or "").strip().lower()
        if status_str == "demanifested":
            continue
            
        if "dlvry phase" in status_str:
            if not has_out_for_delivery:
                dt = item.get("datetime")
                time_str = f" at {_format_datetime(dt)}" if dt else ""
                formatted_history.append(f"Parcel is out for delivery.{time_str}")
                has_out_for_delivery = True
            continue

        if status_str == "delivered" or ("deliver" in status_str and "out" not in status_str and "phase" not in status_str):
            continue
            
        status = item.get("status")
        dt = item.get("datetime")
        reason = item.get("reason")
        if status:
            time_str = f" at {_format_datetime(dt)}" if dt else ""
            elaborated = _elaborate_status(status, reason)
            formatted_history.append(f"{elaborated}{time_str}")

    return {
        "tracking_number": data.get("tracking_no") or data.get("booking_id") or tracking_number,
        "courier": data.get("courier_company_name") or data.get("courier") or "Unknown",
        "customer_name": data.get("customer_name"),
        "email": data.get("customer_email"),
        "phone": data.get("customer_phone"),
        "customer_address": _mask_address(data.get("customer_address")),
        "customer_city": data.get("customer_city"),
        "ordered": _format_datetime(ordered),
        "booking_date": _format_datetime(booking_date_val),
        "status": latest_status,
        "history": formatted_history,
    }



@tool
async def get_order_status(order_id: str) -> dict[str, Any]:
    """Get status, fulfillment, and tracking details for a Shopify order id."""
    normalized = order_id.strip().lstrip("#")
    if not normalized.isdigit():
        return {
            "error": "Shopify Admin REST requires the numeric order id. Ask for the order number or verify via order history first."
        }

    data = await _shopify_get(
        f"orders/{normalized}.json",
        {
            "fields": "id,name,order_number,created_at,financial_status,fulfillment_status,total_price,currency,fulfillments",
        },
    )
    order = data.get("order")
    return {"order": _summarize_order(order)} if order else {"order": None}


@tool
async def get_zahra_order_detail(order_id: str, phone_number: str) -> dict[str, Any]:
    """Get read-only Zahra order invoice details, purchased items, total prices, payment status, or billing details. Requires a numeric order_id and registered phone_number for verification."""
    normalized = order_id.strip().lstrip("#")
    if not normalized.isdigit():
        return {"error": "Please provide a numeric order id."}

    try:
        data = await _zahra_get(f"agent/get-order-detail/{normalized}")
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in {401, 403}:
            return {
                "error": "I couldn't access that order with the current store API access.",
                "status_code": exc.response.status_code,
            }
        if exc.response.status_code == 404:
            return {
                "error": "I couldn't find an order with that order id.",
                "status_code": exc.response.status_code,
            }
        return {
            "error": "The order service returned an error. Please try again shortly.",
            "status_code": exc.response.status_code,
        }
    except httpx.RequestError:
        return {"error": "I couldn't reach the order service right now. Please try again shortly."}

    summary = _summarize_zahra_order_detail(data, normalized)
    
    # Phone Verification Security Check
    actual_phone = summary.get("customer_phone")
    if not actual_phone:
        order_data = data.get("data") if isinstance(data.get("data"), dict) else data
        shipping = order_data.get("shipping_address") or {}
        customer = order_data.get("customer") or {}
        actual_phone = shipping.get("phone") or customer.get("phone")

    if not actual_phone or _normalize_phone(phone_number) != _normalize_phone(actual_phone):
        return {"error": "Verification failed: The phone number provided does not match the order record."}

    return {"order": summary}


@tool
async def track_zahra_booking(tracking_number: str) -> dict[str, Any]:
    """Track a Zahra courier booking by tracking number. Read-only."""
    normalized = tracking_number.strip()
    if not normalized:
        return {"error": "Please provide a tracking number."}

    try:
        data = await _zahra_post("agent/track-booking", {"tracking_number": normalized})
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in {401, 403}:
            return {
                "error": "I couldn't access tracking with the current store API access.",
                "status_code": exc.response.status_code,
            }
        if exc.response.status_code == 404:
            return {
                "error": "I couldn't find tracking details for that number.",
                "status_code": exc.response.status_code,
            }
        return {
            "error": "The tracking service returned an error. Please try again shortly.",
            "status_code": exc.response.status_code,
        }
    except httpx.RequestError:
        return {"error": "I couldn't reach the tracking service right now. Please try again shortly."}

    return {"tracking": _summarize_zahra_tracking(data, normalized)}


@tool
async def track_zahra_order(order_id: str, phone_number: str) -> dict[str, Any]:
    """Track shipping progress, courier delivery status, parcel location, or dispatch timing for a Zahra order. Requires an order_id and registered phone_number for verification."""
    normalized = order_id.strip().lstrip("#")
    if not normalized:
        return {"error": "Please provide an order id."}

    try:
        # Fetch both concurrently using asyncio.gather to minimize latency
        tasks = [
            get_zahra_order_detail.ainvoke({"order_id": normalized, "phone_number": phone_number}),
            _zahra_post("agent/track-order", {"order_id": normalized})
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        order_verify, tracking_res = results[0], results[1]
        
        # 1. Handle phone number verification result first
        if isinstance(order_verify, Exception):
            raise order_verify
        if "error" in order_verify:
            return order_verify  # Returns verification error directly
            
        order_details = order_verify["order"]
        
        # 2. Handle tracking result
        if isinstance(tracking_res, Exception):
            raise tracking_res

        ordered_date = order_details.get("created_at")
        tracking_summary = _summarize_zahra_tracking(
            response=tracking_res,
            tracking_number=normalized,
            ordered=ordered_date,
        )
        return {"tracking": tracking_summary}

    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        # Fallback: verified but not booked yet (tracking API returns error)
        try:
            # We already have order_details verified from the concurrent call
            if 'order_details' in locals():
                return {
                    "not_shipped_yet": True,
                    "order": order_details,
                }
        except Exception:
            pass

        if isinstance(exc, httpx.HTTPStatusError):
            if exc.response.status_code in {401, 403}:
                return {"error": "I couldn't access tracking with the current store API access."}
            if exc.response.status_code == 404:
                return {"error": "I couldn't find tracking details for that order id."}
            return {"error": "The tracking service returned an error. Please try again shortly."}
        return {"error": "I couldn't reach the tracking service right now. Please try again shortly."}


def _summarize_product(product: dict[str, Any]) -> dict[str, Any]:
    variants = product.get("variants") or []
    price = "N/A"
    if variants:
        price = f"{variants[0].get('price')} PKR"
    
    images = product.get("images") or []
    image_url = ""
    if images:
        image_url = images[0].get("src") or ""

    return {
        "id": product.get("id"),
        "title": product.get("title"),
        "handle": product.get("handle"),
        "status": product.get("status"),
        "price": price,
        "image_url": image_url,
    }


@tool
async def get_product_info(product_name_or_id: str) -> dict[str, Any]:
    """Find Shopify product details by product id or title search text."""
    query = product_name_or_id.strip()
    if query.isdigit():
        data = await _shopify_get(f"products/{query}.json")
        product = data.get("product")
        return {"product": _summarize_product(product) if product else None}

    data = await _shopify_get(
        "products.json",
        {
            "title": query,
            "limit": 5,
            "fields": "id,title,handle,status,variants,images",
        },
    )
    products = [_summarize_product(p) for p in data.get("products", [])]
    return {"products": products}


@tool
async def get_all_products(query: str = "") -> dict[str, Any]:
    """Search or list Shopify products and variants."""
    params: dict[str, Any] = {
        "limit": 6,
        "fields": "id,title,handle,status,variants,images",
    }
    if query.strip():
        params["title"] = query.strip()
    data = await _shopify_get("products.json", params)
    products = [_summarize_product(p) for p in data.get("products", [])]
    return {"products": products}


@tool
async def get_categories() -> dict[str, Any]:
    """Get Shopify product categories using custom collections and product types."""
    collections = await _shopify_get(
        "custom_collections.json",
        {"limit": 250, "fields": "id,title,handle,published"},
    )
    products = await _shopify_get(
        "products.json",
        {"limit": 250, "fields": "product_type"},
    )
    product_types = sorted(
        {
            product.get("product_type")
            for product in products.get("products", [])
            if product.get("product_type")
        }
    )
    return {
        "collections": collections.get("custom_collections", []),
        "product_types": product_types,
    }


@tool
async def search_knowledge_base(query: str) -> dict[str, Any]:
    """Search static zahra stores.pk policy, contact, services, and legal pages."""
    return {"results": rag_search_knowledge_base(query)}


ORDER_TOOLS = [
    get_order_status,
    get_zahra_order_detail,
    track_zahra_booking,
    track_zahra_order,
]
PRODUCT_TOOLS = [get_product_info, get_all_products, get_categories]
POLICY_TOOLS = [search_knowledge_base]
ACCOUNT_TOOLS = []
GENERAL_TOOLS = [search_knowledge_base, get_product_info, get_categories]



def get_tools_for_category(category: str):
    if category == "order":
        return ORDER_TOOLS
    if category == "product":
        return PRODUCT_TOOLS
    if category == "policy":
        return POLICY_TOOLS
    if category == "account":
        return ACCOUNT_TOOLS
    return GENERAL_TOOLS
