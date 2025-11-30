import os
import uvicorn
import logging
import json
import asyncio
from datetime import datetime, timedelta
from typing import Tuple, List, Any, AsyncGenerator
from urllib.parse import urlparse
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from truenorth.utils.progress import progress
from truenorth.graph import build_rag_graph, save_graph_as_png
from truenorth.api.routes import pdf
from truenorth.utils.database import init_db, log_request, get_request_count, cleanup_old_logs
from truenorth.utils.citation_manager import CitationManager
from truenorth.agent.state import ChatState

# Get API base URL from environment, default to production
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.mytruenorth.app")

# === Logging Setup ===
file_path = os.path.realpath(__file__)
log_dir = os.path.join(os.path.dirname(file_path), ".logs")
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("server")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if os.getenv("LOG_TO_FILE", "true").lower() == "true":
        log_file = logging.FileHandler(os.path.join(log_dir, "server.log"))
        log_file.setFormatter(formatter)
        logger.addHandler(log_file)

# === Rate Limiting ===
MAX_REQUESTS_PER_DAY = 3


def check_rate_limit(request: Request):
    """
    Check if the client IP has exceeded the daily rate limit using SQLite database.
    Rate limiting is currently disabled.
    """
    # Rate limiting disabled
    return

    # Get client IP (handle proxy headers)
    client_ip = request.headers.get("X-Forwarded-For")
    if client_ip:
        client_ip = client_ip.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"

    # Bypass rate limit for localhost
    if client_ip == "127.0.0.1" or client_ip == "localhost" or client_ip == "::1":
        return

    # Check current count for the last 24 hours
    current_count = get_request_count(client_ip, hours=24)

    # Check if limit reached
    if current_count >= MAX_REQUESTS_PER_DAY:
        logger.warning(f"Rate limit exceeded for IP: {client_ip} (Count: {current_count})")
        raise HTTPException(status_code=429, detail="Daily message limit exceeded (3 messages/day). Please try again tomorrow.")

    # Log the request (consumes one allowance)
    log_request(client_ip)


# === FastAPI App ===
app = FastAPI()


@app.on_event("startup")
async def startup_event():
    init_db()
    # Optional: cleanup logs older than 7 days on startup
    cleanup_old_logs(days=7)


# Register routers
app.include_router(pdf.router)

# Allow UI to make API requests
# Get allowed origins from environment or use defaults
# Broaden for testing
# allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501,http://localhost:3000,http://localhost:3001").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Model Input/Output ===
class QueryInput(BaseModel):
    question: str
    chat_history: List[str] = []
    conversation_id: str = None


class Citation(BaseModel):
    id: int
    author: str
    title: str
    year: str
    page: str
    snippet: str
    url: str
    filename: str


class QueryOutput(BaseModel):
    response: str
    usage: Any
    citations: List[Citation] = []


# Map agent node names to user-friendly status messages
AGENT_STATUS_MESSAGES = {
    "query_router": "Analyzing your question...",
    "web_searcher": "Searching the web for information...",
    "document_retriever": "Searching internal knowledge base...",
    "chitter_chatter": "Engaging in conversation...",
    "query_rewriter": "Refining search query...",
    "evaluate_answer": "Evaluating answer quality...",
    "answer_generator": "Generating comprehensive response...",
    "relevance_grader": "Checking document relevance...",
    "hallucination_checker": "Verifying facts and citations...",
}


# === Core Functionality ===
async def stream_workflow(question: str) -> AsyncGenerator[str, None]:
    """
    Streams workflow events and the final response as SSE.
    """
    logger.info(f"Starting streaming workflow for: {question}")

    model_name = "gemini-2.0-flash"
    model_provider = "Gemini"
    selected_analysts = []

    workflow = build_rag_graph(selected_analysts)
    agent = workflow.compile()

    inputs = {
        "question": question,
        "messages": [HumanMessage(content=question)],
        "data": [],
        "metadata": {
            "show_reasoning": True,
            "model_name": model_name,
            "model_provider": model_provider,
        },
    }

    try:
        # Stream events from the graph
        async for event in agent.astream_events(inputs, version="v2"):
            kind = event["event"]

            # Handle node start events to send status updates
            if kind == "on_chain_start":
                name = event["name"]
                if name in AGENT_STATUS_MESSAGES:
                    status_msg = AGENT_STATUS_MESSAGES[name]
                    yield f"data: {json.dumps({'type': 'status', 'message': status_msg})}\n\n"

            # Handle final output
            # The final output comes from on_chain_end of the compiled graph (LangGraph)
            # But it's easier to just check for the final state in the loop or return it at end
            pass

    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        return

    # Getting final result (re-running invoke is safer to get full consistent state)
    # Note: In production, you'd want to capture state from the stream loop to avoid re-execution.
    final_state_dict = await agent.ainvoke(inputs)

    # Cast back to ChatState object for CitationManager
    final_state = ChatState(**final_state_dict)

    # Process final response
    response_text = final_state.generation or str(final_state)

    # Extract citations using CitationManager and get renumbered text
    # This replaces the entire block of manual metadata extraction
    citations, renumbered_response_text = CitationManager.resolve_citations(final_state)

    # Yield final result
    result = {"type": "result", "response": renumbered_response_text, "citations": citations, "conversation_id": "default"}
    yield f"data: {json.dumps(result)}\n\n"
    yield "data: [DONE]\n\n"


# Kept for backward compatibility
async def invoke_llm(question: str) -> Tuple[str, List[Any], List[Citation]]:
    logger.info("Invoking chatbot (legacy invoke)...")
    progress.start()
    try:
        model_name = "gemini-2.0-flash"
        model_provider = "Gemini"
        selected_analysts = []

        workflow = build_rag_graph(selected_analysts)
        agent = workflow.compile()

        final_state_dict = await agent.ainvoke(
            {
                "question": question,
                "messages": [HumanMessage(content=question)],
                "data": [],
                "metadata": {
                    "show_reasoning": True,
                    "model_name": model_name,
                    "model_provider": model_provider,
                },
            }
        )

        # Cast back to ChatState object for CitationManager
        final_state = ChatState(**final_state_dict)

        response = final_state.generation

        # Use CitationManager for legacy endpoint too
        raw_citations, renumbered_response = CitationManager.resolve_citations(final_state)
        # Convert dicts back to Pydantic models for legacy return signature
        citations = [Citation(**c) for c in raw_citations]

        return renumbered_response, final_state, citations

    finally:
        progress.stop()


@app.post("/query", response_model=QueryOutput)
async def get_chat_response(input_data: QueryInput, request: Request):
    # Enforce rate limit
    check_rate_limit(request)

    response, final_state, citations = await invoke_llm(input_data.question)
    return QueryOutput(response=response, usage={}, citations=citations)


@app.post("/api/chat/stream")
async def stream_chat_response(request: Request):
    """
    Streaming endpoint for chat responses.
    Returns SSE with status updates and final result.
    """
    # Enforce rate limit
    check_rate_limit(request)

    try:
        data = await request.json()
        input_data = QueryInput(**data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    return StreamingResponse(stream_workflow(input_data.question), media_type="text/event-stream")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
