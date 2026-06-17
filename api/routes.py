from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.graph import run_agent, stream_agent


router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_token: str = ""
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    response: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    response = await run_agent(
        message=request.message,
        session_token=request.session_token,
        conversation_history=request.conversation_history,
    )
    return ChatResponse(response=response)


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_agent(
            message=request.message,
            session_token=request.session_token,
            conversation_history=request.conversation_history,
        ),
        media_type="text/plain; charset=utf-8",
    )

