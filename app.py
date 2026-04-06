import os
import sys
import platform
import subprocess
import requests
import time
import json
import traceback
import re
import logging
import inspect
import asyncio
import operator
import base64
from contextlib import asynccontextmanager
from typing import Tuple, List, Dict, Any, Optional, TypeVar, Type, Union, Literal
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse, quote, quote_plus
from datetime import datetime, timedelta

# Third-party imports
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from colorama import Fore, Style
import questionary
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.style import Style as RichStyle
from rich.text import Text
import uvicorn
import fitz  # PyMuPDF

# FastAPI
from fastapi import FastAPI, Request, HTTPException, APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# LangChain & LangGraph
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_groq import ChatGroq
# OpenAI imports are lazy-loaded only when needed to avoid unnecessary API validation
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, messages_from_dict, messages_to_dict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain.load import dumps, loads
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledGraph
from sqlalchemy import JSON, Boolean, DateTime, Integer, String, Text, create_engine, func, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

# --- Environment & Logging Setup ---

load_dotenv()

def get_caller_logger(to_stdout: bool = True) -> logging.Logger:
    caller_frame = inspect.stack()[1]
    module = inspect.getmodule(caller_frame[0])
    logger_name = module.__name__ if module else "__main__"
    file_path = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(file_path, ".logs")
    log_file = os.path.join(log_dir, "agent_pipeline.log")

    # Create .logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(funcName)s] %(message)s")

    # Prevent adding duplicate handlers
    if not logger.handlers:
        # File handler
        if os.getenv("LOG_TO_FILE", "true").lower() == "true":
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Optional stdout handler
        if to_stdout:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

    return logger

logger = get_caller_logger()

# --- Configuration & Constants ---

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.mytruenorth.app")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
VECTOR_STORE_PATH = Path(__file__).parent/ "vector_store" / "truenorth_kb_vectorstore/"
BOOKS_DIR = Path(__file__).parent / "books_pdf"
OLLAMA_SERVER_URL = "http://localhost:11434"
OLLAMA_API_MODELS_ENDPOINT = f"{OLLAMA_SERVER_URL}/api/tags"
OLLAMA_DOWNLOAD_URL = {
    "darwin": "https://ollama.com/download/darwin", 
    "win32": "https://ollama.com/download/windows", 
    "linux": "https://ollama.com/download/linux"
}
POSTGRES_DEFAULT_HOST = "postgres" if Path("/.dockerenv").exists() else "localhost"
POSTGRES_HOST = os.getenv("POSTGRES_HOST", POSTGRES_DEFAULT_HOST)
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_USER = os.getenv("POSTGRES_USER", "truenorth")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "truenorth")
CHAT_STATE_DB_NAME = os.getenv("CHAT_STATE_DB_NAME", "truenorth_chat")
CHAT_LOG_DB_NAME = os.getenv("CHAT_LOG_DB_NAME", "truenorth_logs")
DB_CONNECT_RETRIES = int(os.getenv("DB_CONNECT_RETRIES", "10"))
DB_CONNECT_RETRY_DELAY_SECONDS = float(os.getenv("DB_CONNECT_RETRY_DELAY_SECONDS", "2"))
THINK_MODEL_NAME = os.getenv("MODEL_NAME_THINK", "models/gemini-3.1-pro-preview")
if not os.path.exists(VECTOR_STORE_PATH):
    logger.warning(f"Vector store directory not found at {VECTOR_STORE_PATH}. Please ensure it exists.")
if not os.path.exists(BOOKS_DIR):
    logger.warning(f"Books directory not found at {BOOKS_DIR}. Please ensure it exists and contains the necessary PDF files.")

# --- Models & Enums ---

class ModelProvider(str, Enum):
    """Enum for supported LLM providers"""
    ANTHROPIC = "Anthropic"
    DEEPSEEK = "DeepSeek"
    GEMINI = "Gemini"
    GROQ = "Groq"
    OPENAI = "OpenAI"
    OLLAMA = "Ollama"

class LLMModel(BaseModel):
    """Represents an LLM model configuration"""
    display_name: str
    model_name: str
    provider: ModelProvider

    def to_choice_tuple(self) -> Tuple[str, str, str]:
        """Convert to format needed for questionary choices"""
        return (self.display_name, self.model_name, self.provider.value)

    def has_json_mode(self) -> bool:
        """Check if the model supports JSON mode"""
        if self.is_deepseek() or self.is_gemini():
            return False
        if self.is_ollama():
            return "llama3" in self.model_name or "neural-chat" in self.model_name
        return True

    def is_deepseek(self) -> bool:
        return self.model_name.startswith("deepseek")

    def is_gemini(self) -> bool:
        return self.model_name.startswith("gemini")

    def is_ollama(self) -> bool:
        return self.provider == ModelProvider.OLLAMA

AVAILABLE_MODELS = [
    LLMModel(display_name="[anthropic] claude-3.5-haiku", model_name="claude-3-5-haiku-latest", provider=ModelProvider.ANTHROPIC),
    LLMModel(display_name="[anthropic] claude-3.5-sonnet", model_name="claude-3-5-sonnet-latest", provider=ModelProvider.ANTHROPIC),
    LLMModel(display_name="[anthropic] claude-3.7-sonnet", model_name="claude-3-7-sonnet-latest", provider=ModelProvider.ANTHROPIC),
    LLMModel(display_name="[deepseek] deepseek-r1", model_name="deepseek-reasoner", provider=ModelProvider.DEEPSEEK),
    LLMModel(display_name="[deepseek] deepseek-v3", model_name="deepseek-chat", provider=ModelProvider.DEEPSEEK),
    LLMModel(display_name="[gemini] gemini-2.0-flash", model_name="gemini-2.0-flash", provider=ModelProvider.GEMINI),
    LLMModel(display_name="[gemini] gemini-2.5-flash", model_name="gemini-2.5-flash", provider=ModelProvider.GEMINI),
    LLMModel(display_name="[gemini] gemini-2.5-pro", model_name="gemini-2.5-pro-exp-03-25", provider=ModelProvider.GEMINI),
    LLMModel(display_name="[gemini] gemini-3-pro-preview", model_name="gemini-3-pro-preview", provider=ModelProvider.GEMINI),
    LLMModel(display_name="[groq] llama-4-scout-17b", model_name="meta-llama/llama-4-scout-17b-16e-instruct", provider=ModelProvider.GROQ),
    LLMModel(display_name="[groq] llama-4-maverick-17b", model_name="meta-llama/llama-4-maverick-17b-128e-instruct", provider=ModelProvider.GROQ),
    LLMModel(display_name="[openai] gpt-5", model_name="gpt-5", provider=ModelProvider.OPENAI),
    LLMModel(display_name="[openai] gpt-5-mini", model_name="gpt-5-mini", provider=ModelProvider.OPENAI),
    LLMModel(display_name="[openai] gpt-5-nano", model_name="gpt-5-nano", provider=ModelProvider.OPENAI),
    LLMModel(display_name="[openai] gpt-4.5", model_name="gpt-4.5-preview", provider=ModelProvider.OPENAI),
    LLMModel(display_name="[openai] gpt-4o", model_name="gpt-4o", provider=ModelProvider.OPENAI),
    LLMModel(display_name="[openai] o1", model_name="o1", provider=ModelProvider.OPENAI),
    LLMModel(display_name="[openai] o3-mini", model_name="o3-mini", provider=ModelProvider.OPENAI),
]

OLLAMA_MODELS = [
    LLMModel(display_name="[ollama] smollm (1.7B)", model_name="smollm:1.7b", provider=ModelProvider.OLLAMA),
    LLMModel(display_name="[ollama] phi3  (3.8B)", model_name="phi3", provider=ModelProvider.OLLAMA),
    LLMModel(display_name="[ollama] gemma3 (4B)", model_name="gemma3:4b", provider=ModelProvider.OLLAMA),
    LLMModel(display_name="[ollama] qwen2.5 (7B)", model_name="qwen2.5", provider=ModelProvider.OLLAMA),
    LLMModel(display_name="[ollama] llama3.1 (8B)", model_name="llama3.1:latest", provider=ModelProvider.OLLAMA),
    LLMModel(display_name="[ollama] gemma3 (12B)", model_name="gemma3:12b", provider=ModelProvider.OLLAMA),
    LLMModel(display_name="[ollama] mistral-small3.1 (24B)", model_name="mistral-small3.1", provider=ModelProvider.OLLAMA),
    LLMModel(display_name="[ollama] gemma3 (27B)", model_name="gemma3:27b", provider=ModelProvider.OLLAMA),
    LLMModel(display_name="[ollama] qwen2.5 (32B)", model_name="qwen2.5:32b", provider=ModelProvider.OLLAMA),
    LLMModel(display_name="[ollama] llama-3.3 (70B)", model_name="llama3.3:70b-instruct-q4_0", provider=ModelProvider.OLLAMA),
]

LLM_ORDER = [model.to_choice_tuple() for model in AVAILABLE_MODELS]
OLLAMA_LLM_ORDER = [model.to_choice_tuple() for model in OLLAMA_MODELS]


# --- Prompts & System Strings ---
DP1 = "engineer an emotionally intelligent and stereotype-neutral interface within the conversational agent to encourage contextually relevant and engaging interactions, allowing the user to develop high competency and reputation in professional and academic STEM environments"
DP2 = "develop a trustworthy and personalized learning environment through the conversational agent that fosters a sense of local and virtual community among users and fellow STEM colleagues, fostering a sense of community among users to support individualized and high-achieving educational and professional experiences."
DP3 = "facilitate empowering and streamlined interactions with the conversational agent through simplified dialogues to bolster comprehension and foster independence for users in professional and academic STEM environments."

goals_as_str = "\n".join([f"{i}. {goal}" for i, goal in enumerate([DP1, DP2, DP3])])
goals_as_str += "\n If relevant, PERMA+4 are pillars of wellbeing: Positive Emotions, Engagement, Relationships, Meaning, Accomplishment, Physical Health, Positive Mindset, Environment, and Economic Security."

vectorstore_content_summary = "workplace wellbeing, communcation strategies rooted in conflict resolution and diplomacy, positive psychology, leadership skills, coping mechanisms"
system_relevant_scope = "technical or engineering knowledge, software development, management and leadership, optimizing comfort in lived environment, maintaining positive trajectory towards maximizing STEM career"

# --- Common Data Structures ---

class SimpleCitation(BaseModel):
    id: int
    author: str
    title: str
    year: str
    page: str
    snippet: str
    url: str
    filename: str

class CitationSource(BaseModel):
    """Internal representation of a available source (PDF or Web)."""
    source_id: int  # Immutable ID (1, 2, 3...)
    type: str  # "pdf" or "web"

    # Display Metadata
    author: str
    title: str
    year: str
    page: str = ""
    url: str  # API endpoint (PDF) or Web URL
    filename: str  # For icon logic
    quote: str = "" # The exact text extracted by LLM

    # Content
    content_snippet: str  # Fallback content (~200 chars)
    full_content: str = Field(exclude=True)  # Full text for LLM context (excluded from serialization if needed)

class CitedSource(BaseModel):
    """Represents a citation used in the generated answer."""
    source_id: int
    quote: str = Field(description="The exact quote from the source that supports the claim.")

class ChatState(BaseModel):
    snowflake: str = ''
    anonymized_history_consent: bool = False

    question: str = ""
    original_question: Optional[str] = None
    generation: Optional[str] = None

    messages: list = Field(default_factory=list)
    current_user_message: str | None = None
    documents: List[Any] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    citation_registry: Dict[int, CitationSource] = Field(default_factory=dict)
    generated_citations: List[CitedSource] = Field(default_factory=list)

    # Intent tracking
    user_intent: Optional[str] = None  # Extracted user intent
    intent_metadata: Dict[str, Any] = Field(default_factory=dict)  # Intent details

    max_retries: int = 2
    current_try: int = 0

    def add_user_message(self, content: str, anonymized_history_consent: Optional[bool] = None):
        additional_kwargs = {}
        if anonymized_history_consent is not None:
            additional_kwargs["anonymized_history_consent"] = anonymized_history_consent
        self.messages.append(HumanMessage(content=content, additional_kwargs=additional_kwargs))

    def add_agent_message(self, content: str, anonymized_history_consent: Optional[bool] = None):
        additional_kwargs = {}
        if anonymized_history_consent is not None:
            additional_kwargs["anonymized_history_consent"] = anonymized_history_consent
        self.messages.append(AIMessage(content=content, additional_kwargs=additional_kwargs))

    def clear_conversation(self):
        self.messages.clear()

def build_postgres_url(database_name: str, env_var: str) -> str:
    default_url = (
        f"postgresql+psycopg2://{quote_plus(POSTGRES_USER)}:{quote_plus(POSTGRES_PASSWORD)}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{database_name}"
    )
    return os.getenv(env_var, default_url)


POSTGRES_ADMIN_DB_NAME = os.getenv("POSTGRES_ADMIN_DB_NAME", "postgres")
CHAT_STATE_DATABASE_URL = build_postgres_url(CHAT_STATE_DB_NAME, "CHAT_STATE_DATABASE_URL")
CHAT_LOG_DATABASE_URL = build_postgres_url(CHAT_LOG_DB_NAME, "CHAT_LOG_DATABASE_URL")
POSTGRES_ADMIN_DATABASE_URL = build_postgres_url(POSTGRES_ADMIN_DB_NAME, "POSTGRES_ADMIN_DATABASE_URL")


class ChatStateStorageBase(DeclarativeBase):
    pass


class ChatLogStorageBase(DeclarativeBase):
    pass


class ChatStateRecord(ChatStateStorageBase):
    __tablename__ = "chat_states"

    snowflake: Mapped[str] = mapped_column(String(255), primary_key=True)
    state: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class ChatHistoryLog(ChatLogStorageBase):
    __tablename__ = "chat_history_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snowflake: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    conversation_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    anonymized_history_consent: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


chat_state_engine = create_engine(
    CHAT_STATE_DATABASE_URL,
    future=True,
    pool_pre_ping=True,
    connect_args={"connect_timeout": 5},
)
chat_log_engine = create_engine(
    CHAT_LOG_DATABASE_URL,
    future=True,
    pool_pre_ping=True,
    connect_args={"connect_timeout": 5},
)
postgres_admin_engine = create_engine(
    POSTGRES_ADMIN_DATABASE_URL,
    future=True,
    pool_pre_ping=True,
    connect_args={"connect_timeout": 5},
    isolation_level="AUTOCOMMIT",
)
ChatStateSessionLocal = sessionmaker(bind=chat_state_engine, autoflush=False, expire_on_commit=False)
ChatLogSessionLocal = sessionmaker(bind=chat_log_engine, autoflush=False, expire_on_commit=False)
POSTGRES_STORAGE_READY = False


def serialize_document(document: Any) -> dict:
    if isinstance(document, Document):
        return document.model_dump()
    if isinstance(document, dict):
        return document
    return {
        "page_content": getattr(document, "page_content", str(document)),
        "metadata": getattr(document, "metadata", {}),
    }


def deserialize_document(document_data: Any) -> Document:
    if isinstance(document_data, Document):
        return document_data
    if isinstance(document_data, dict):
        return Document(**document_data)
    return Document(page_content=str(document_data), metadata={})


def deserialize_messages(message_data: List[dict]) -> List[BaseMessage]:
    if not message_data:
        return []

    try:
        return messages_from_dict(message_data)
    except Exception:
        restored_messages: List[BaseMessage] = []
        for item in message_data:
            if not isinstance(item, dict):
                continue

            msg_type = item.get("type")
            if msg_type == "human":
                restored_messages.append(HumanMessage(**item))
            elif msg_type == "ai":
                restored_messages.append(AIMessage(**item))
            elif msg_type == "system":
                restored_messages.append(SystemMessage(**item))

        return restored_messages


def serialize_chat_state(state: ChatState) -> dict:
    return {
        "snowflake": state.snowflake,
        "anonymized_history_consent": state.anonymized_history_consent,
        "question": state.question,
        "original_question": state.original_question,
        "generation": state.generation,
        "messages": messages_to_dict(state.messages),
        "current_user_message": state.current_user_message,
        "documents": [serialize_document(doc) for doc in state.documents],
        "metadata": state.metadata,
        "citation_registry": {str(key): value.model_dump() for key, value in state.citation_registry.items()},
        "generated_citations": [citation.model_dump() for citation in state.generated_citations],
        "user_intent": state.user_intent,
        "intent_metadata": state.intent_metadata,
        "max_retries": state.max_retries,
        "current_try": state.current_try,
    }


def deserialize_chat_state(state_payload: Optional[dict], user_snowflake: str) -> ChatState:
    if not state_payload:
        return ChatState(snowflake=user_snowflake)

    raw_citation_registry = state_payload.get("citation_registry", {}) or {}
    citation_registry = {
        int(key): CitationSource(**value)
        for key, value in raw_citation_registry.items()
        if isinstance(value, dict)
    }

    generated_citations = [
        CitedSource(**citation)
        for citation in state_payload.get("generated_citations", [])
        if isinstance(citation, dict)
    ]

    return ChatState(
        snowflake=state_payload.get("snowflake", user_snowflake),
        anonymized_history_consent=bool(state_payload.get("anonymized_history_consent", False)),
        question=state_payload.get("question", ""),
        original_question=state_payload.get("original_question"),
        generation=state_payload.get("generation"),
        messages=deserialize_messages(state_payload.get("messages", [])),
        current_user_message=state_payload.get("current_user_message"),
        documents=[deserialize_document(doc) for doc in state_payload.get("documents", [])],
        metadata=state_payload.get("metadata", {}) or {},
        citation_registry=citation_registry,
        generated_citations=generated_citations,
        user_intent=state_payload.get("user_intent"),
        intent_metadata=state_payload.get("intent_metadata", {}) or {},
        max_retries=state_payload.get("max_retries", 2),
        current_try=state_payload.get("current_try", 0),
    )


def ensure_database_exists(database_name: str) -> None:
    if not re.match(r"^[A-Za-z0-9_]+$", database_name):
        raise ValueError(f"Invalid Postgres database name: {database_name}")

    with postgres_admin_engine.connect() as connection:
        exists = connection.exec_driver_sql(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (database_name,),
        ).scalar()
        if exists:
            return

        connection.exec_driver_sql(f'CREATE DATABASE "{database_name}"')
        logger.info("Created Postgres database '%s'", database_name)


def ensure_chat_log_schema() -> None:
    """Ensure chat_history_logs table has all required columns and indexes"""
    with chat_log_engine.begin() as connection:
        # Add missing column if needed
        connection.exec_driver_sql(
            """
            ALTER TABLE chat_history_logs
            ADD COLUMN IF NOT EXISTS anonymized_history_consent BOOLEAN NOT NULL DEFAULT FALSE
            """
        )
        
        # Create indexes for efficient querying
        connection.exec_driver_sql(
            """
            CREATE INDEX IF NOT EXISTS ix_chat_history_logs_anonymized_history_consent
            ON chat_history_logs (anonymized_history_consent)
            """
        )
        connection.exec_driver_sql(
            """
            CREATE INDEX IF NOT EXISTS ix_chat_history_logs_conversation_id
            ON chat_history_logs (conversation_id)
            """
        )
        connection.exec_driver_sql(
            """
            CREATE INDEX IF NOT EXISTS ix_chat_history_logs_snowflake
            ON chat_history_logs (snowflake)
            """
        )
        connection.exec_driver_sql(
            """
            CREATE INDEX IF NOT EXISTS ix_chat_history_logs_created_at
            ON chat_history_logs (created_at DESC)
            """
        )
        connection.exec_driver_sql(
            """
            CREATE INDEX IF NOT EXISTS ix_chat_history_logs_snowflake_created_at
            ON chat_history_logs (snowflake, created_at DESC)
            """
        )
        connection.exec_driver_sql(
            """
            CREATE INDEX IF NOT EXISTS ix_chat_history_logs_conversation_created_at
            ON chat_history_logs (conversation_id, created_at ASC)
            """
        )
        
        logger.info("Chat history logs schema and indexes ensured")


def initialize_postgres_storage() -> None:
    global POSTGRES_STORAGE_READY

    if POSTGRES_STORAGE_READY:
        return

    last_error: Exception | None = None
    for attempt in range(1, DB_CONNECT_RETRIES + 1):
        try:
            ensure_database_exists(CHAT_STATE_DB_NAME)
            ensure_database_exists(CHAT_LOG_DB_NAME)

            with chat_state_engine.connect() as connection:
                connection.exec_driver_sql("SELECT 1")
            with chat_log_engine.connect() as connection:
                connection.exec_driver_sql("SELECT 1")

            ChatStateStorageBase.metadata.create_all(chat_state_engine)
            ChatLogStorageBase.metadata.create_all(chat_log_engine)
            ensure_chat_log_schema()
            POSTGRES_STORAGE_READY = True
            logger.info(
                "Postgres storage ready for chat state DB '%s' and logs DB '%s'",
                CHAT_STATE_DB_NAME,
                CHAT_LOG_DB_NAME,
            )
            return
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Postgres initialization attempt %s/%s failed: %s",
                attempt,
                DB_CONNECT_RETRIES,
                exc,
            )
            if attempt < DB_CONNECT_RETRIES:
                time.sleep(DB_CONNECT_RETRY_DELAY_SECONDS)

    raise RuntimeError("Unable to initialize Postgres storage") from last_error


def get_state(user_snowflake: str) -> ChatState:
    initialize_postgres_storage()

    with ChatStateSessionLocal() as session:
        record = session.get(ChatStateRecord, user_snowflake)
        if record is None:
            state = ChatState(snowflake=user_snowflake)
            session.add(ChatStateRecord(snowflake=user_snowflake, state=serialize_chat_state(state)))
            session.commit()
            return state

        return deserialize_chat_state(record.state, user_snowflake)


def save_state(state: ChatState) -> None:
    initialize_postgres_storage()
    payload = serialize_chat_state(state)

    with ChatStateSessionLocal() as session:
        record = session.get(ChatStateRecord, state.snowflake)
        if record is None:
            session.add(ChatStateRecord(snowflake=state.snowflake, state=payload))
        else:
            record.state = payload
        session.commit()


def log_chat_history(
    snowflake: str,
    role: str,
    content: str,
    conversation_id: Optional[str] = None,
    anonymized_history_consent: bool = False,
    payload: Optional[dict] = None,
) -> None:
    initialize_postgres_storage()
    payload_data = dict(payload or {})
    payload_data["anonymized_history_consent"] = anonymized_history_consent

    with ChatLogSessionLocal() as session:
        session.add(
            ChatHistoryLog(
                snowflake=snowflake,
                conversation_id=conversation_id,
                role=role,
                content=content,
                anonymized_history_consent=anonymized_history_consent,
                payload=payload_data,
            )
        )
        session.commit()


# --- Pipeline Log Parser ---

def parse_pipeline_logs(log_file_path: str = None, limit: int = 100) -> List["PipelineExecution"]:
    """
    Parse agent_pipeline.log file to extract pipeline execution records.
    Returns list of PipelineExecution objects ordered by timestamp (newest first).
    """
    if log_file_path is None:
        file_path = os.path.dirname(os.path.realpath(__file__))
        log_file_path = os.path.join(file_path, ".logs", "agent_pipeline.log")
    
    if not os.path.exists(log_file_path):
        logger.warning(f"Pipeline log file not found: {log_file_path}")
        return []
    
    executions = []
    current_execution = None
    current_stages = []
    stage_start_times = {}
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Parse timestamp and message
                match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[(\w+)\] \[(\w+)\] (.+)', line)
                if not match:
                    continue
                
                timestamp_str, level, function, message = match.groups()
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                
                # Workflow execution start
                if '🚀 WORKFLOW EXECUTION START' in message:
                    if current_execution:
                        # Save previous execution
                        current_execution.stages = current_stages
                        executions.append(current_execution)
                    
                    current_execution = PipelineExecution(
                        id=timestamp.isoformat(),
                        snowflake='',
                        question='',
                        start_time=timestamp.isoformat(),
                        status='running',
                        agent_sequence=[],
                        stages=[]
                    )
                    current_stages = []
                    stage_start_times = {}
                
                # Extract workflow metadata
                elif current_execution:
                    if 'Question:' in message:
                        current_execution.question = message.split('Question:', 1)[1].strip()
                    elif 'Snowflake:' in message:
                        current_execution.snowflake = message.split('Snowflake:', 1)[1].strip()
                    
                    # Agent started
                    elif '▶️  Agent started:' in message:
                        agent_name = message.split('Agent started:', 1)[1].strip()
                        current_execution.agent_sequence.append(agent_name)
                        stage_start_times[agent_name] = timestamp
                        
                        # Add stage record
                        current_stages.append(PipelineStage(
                            agent_name=agent_name,
                            status='started',
                            start_time=timestamp.isoformat(),
                            status_message=f"Starting {agent_name}..."
                        ))
                    
                    # Agent completed
                    elif '✓ Agent completed:' in message:
                        agent_name = message.split('Agent completed:', 1)[1].strip()
                        if agent_name in stage_start_times:
                            start = stage_start_times[agent_name]
                            duration_ms = int((timestamp - start).total_seconds() * 1000)
                            
                            # Update last stage
                            for stage in reversed(current_stages):
                                if stage.agent_name == agent_name and stage.status == 'started':
                                    stage.status = 'completed'
                                    stage.end_time = timestamp.isoformat()
                                    stage.duration_ms = duration_ms
                                    stage.status_message = f"Completed in {duration_ms}ms"
                                    break
                    
                    # Workflow summary
                    elif '📊 Workflow Summary:' in message:
                        # Next lines contain duration and agent sequence
                        pass
                    elif current_execution.status == 'running' and 'Duration:' in message:
                        duration_str = message.split('Duration:', 1)[1].strip()
                        # Parse "4.05s" format
                        if duration_str.endswith('s'):
                            duration_sec = float(duration_str[:-1])
                            current_execution.duration_ms = int(duration_sec * 1000)
                    
                    # Workflow complete
                    elif '✅ WORKFLOW EXECUTION COMPLETE' in message:
                        if current_execution:
                            current_execution.status = 'success'
                            current_execution.end_time = timestamp.isoformat()
                            current_execution.stages = current_stages
                            executions.append(current_execution)
                            current_execution = None
                            current_stages = []
                            stage_start_times = {}
                    
                    # Error detection
                    elif level == 'ERROR' and current_execution:
                        current_execution.status = 'error'
                        current_execution.error_message = message
        
        # Handle last execution if file ended without completion
        if current_execution:
            current_execution.stages = current_stages
            executions.append(current_execution)
    
    except Exception as e:
        logger.error(f"Error parsing pipeline logs: {e}")
        logger.error(traceback.format_exc())
    
    # Return newest first, limited
    executions.reverse()
    return executions[:limit]


# --- LLM Utils ---

def get_model_info(model_name: str) -> Optional[LLMModel]:
    return next((m for m in AVAILABLE_MODELS + OLLAMA_MODELS if m.model_name == model_name), None)

def get_model(model_name: str, model_provider: ModelProvider):
    logger.info(f"Model name: {model_name}")
    logger.info(f"Model provider: {model_provider}")
    
    if model_provider == ModelProvider.GROQ:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure GROQ_API_KEY is set in your .env file.")
            raise ValueError("Groq API key not found.")
        return ChatGroq(model=model_name, api_key=api_key)
    elif model_provider == ModelProvider.OPENAI:
        # Lazy-load OpenAI to avoid unnecessary API validation at startup
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure OPENAI_API_KEY is set in your .env file.")
            raise ValueError("OpenAI API key not found.")
        return ChatOpenAI(model=model_name, api_key=api_key)
    elif model_provider == ModelProvider.ANTHROPIC:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure ANTHROPIC_API_KEY is set in your .env file.")
            raise ValueError("Anthropic API key not found.")
        return ChatAnthropic(model=model_name, api_key=api_key)
    elif model_provider == ModelProvider.DEEPSEEK:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure DEEPSEEK_API_KEY is set in your .env file.")
            raise ValueError("DeepSeek API key not found.")
        return ChatDeepSeek(model=model_name, api_key=api_key)
    elif model_provider == ModelProvider.GEMINI:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure GEMINI_API_KEY is set in your .env file.")
            raise ValueError("Google API key not found.")
        return ChatGoogleGenerativeAI(model=model_name, api_key=api_key)
    elif model_provider == ModelProvider.OLLAMA:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model_name, base_url=base_url)

T = TypeVar("T", bound=BaseModel)

def is_root_model(model_class: Type[BaseModel]) -> bool:
    return list(model_class.model_fields.keys()) == ["root"]

def instantiate_model(model_class: Type[T], data: Any) -> T:
    if is_root_model(model_class):
        return model_class(root=data)
    elif isinstance(data, dict):
        return model_class(**data)
    else:
        field = list(model_class.model_fields.keys())[0]
        return model_class(**{field: data})

def extract_json_from_response(text: Union[str, bytes]) -> Optional[dict]:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")

    # 1. Try standard Markdown JSON block
    md_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if md_block:
        try:
            return json.loads(md_block.group(1))
        except json.JSONDecodeError:
            pass

    # 2. Try finding the first brace-enclosed object
    json_candidates = re.findall(r"(\{.*?\})", text, re.DOTALL)
    for candidate in json_candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # 3. Try parsing the whole text
    try:
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return json.loads(cleaned.strip())
    except json.JSONDecodeError:
        pass

    return None

def create_default_response(model_class: Optional[Type[T]]) -> Optional[T]:
    if model_class is None:
        return None
    try:
        return instantiate_model(model_class, False)
    except Exception:
        default_fields = {k: ("Error" if v.annotation == str else 0 if v.annotation in (int, float) else {} if v.annotation == dict else None) for k, v in model_class.model_fields.items()}
        return model_class(**default_fields)

def call_llm(prompt: Any, model_name: str, model_provider: str, pydantic_model: Type[T], agent_name: Optional[str] = None, max_retries: int = 3, default_factory=None, verbose=False) -> T:
    model_info = get_model_info(model_name)
    llm = get_model(model_name, model_provider)

    log_prefix = f"[{agent_name}] " if agent_name else ""
    logger.info(f"{log_prefix}🤖 LLM Call: {model_provider}/{model_name}")
    
    if pydantic_model:
        logger.info(f"{log_prefix}Expected output: {pydantic_model.__name__}")

    if llm is None:
        logger.error(f"{log_prefix}❌ Failed to instantiate model: {model_name} from {model_provider}")
        return default_factory() if default_factory else create_default_response(pydantic_model)

    if pydantic_model:
        if model_info or model_provider in ["OpenAI", "Anthropic", "Gemini"]:
            logger.info(f"{log_prefix}Configuring structured output...")
            llm = llm.with_structured_output(pydantic_model)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"{log_prefix}API call attempt {attempt}/{max_retries}...")
            if verbose:
                logger.debug(f"{log_prefix}Prompt: {prompt}")

            start_time = time.time()
            result = llm.invoke(prompt)
            duration = time.time() - start_time
            
            logger.info(f"{log_prefix}✓ API response received in {duration:.2f}s")

            if verbose:
                logger.debug(f"{log_prefix}Result type: {type(result)}")
                if not isinstance(result, pydantic_model):
                    logger.debug(f"{log_prefix}Result content: {result}")

            if pydantic_model and isinstance(result, pydantic_model):
                logger.info(f"{log_prefix}✓ Successfully parsed as {pydantic_model.__name__}")
                return result

            # Fallback Handling
            if pydantic_model:
                logger.info(f"{log_prefix}Attempting fallback parsing...")
                result_content = ""
                if isinstance(result, dict):
                    return instantiate_model(pydantic_model, result)
                elif hasattr(result, "content"):
                    result_content = str(result.content)
                else:
                    result_content = str(result)

                if hasattr(result, "tool_calls") and result.tool_calls:
                    try:
                        args = result.tool_calls[0]["args"]
                        logger.info(f"{log_prefix}Parsing from tool_calls...")
                        return instantiate_model(pydantic_model, args)
                    except Exception as e:
                        logger.warning(f"{log_prefix}Failed to parse tool_calls: {e}")

                parsed_json = extract_json_from_response(result_content)
                if parsed_json:
                    logger.info(f"{log_prefix}✓ Extracted JSON from response")
                    return instantiate_model(pydantic_model, parsed_json)

                # Gemini Fallback mainly
                if model_provider == "Gemini" or (model_info and model_info.is_gemini()):
                    try:
                        fields = list(pydantic_model.model_fields.keys())
                        if "answer" in fields and "citations" in fields:
                            logger.info(f"{log_prefix}Applying Gemini heuristic parsing...")
                            # Heuristic extraction for AnswerResponse
                            answer_text = result_content
                            citations = []
                            split_patterns = [r"###\s*Citations", r"\*\*\s*References\s*\*\*", r"###\s*References"]
                            for pattern in split_patterns:
                                parts = re.split(pattern, result_content, flags=re.IGNORECASE)
                                if len(parts) > 1:
                                    answer_text = parts[0].strip()
                                    citation_text = parts[1].strip()
                                    citation_matches = re.findall(r'\[(\d+)\]\s*"(.*?)"', citation_text, re.DOTALL)
                                    for source_id, quote in citation_matches:
                                        citations.append({"source_id": int(source_id), "quote": quote.strip()})
                                    break
                            
                            return instantiate_model(pydantic_model, {"answer": answer_text, "citations": citations})
                        elif "content" in fields:
                             return instantiate_model(pydantic_model, {"content": result_content})
                    except Exception as e:
                         logger.warning(f"{log_prefix}Fallback parsing failed: {e}")

                raise ValueError(f"Failed to parse structured output from response: {result_content[:200]}...")

            return result

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"{log_prefix}❌ LLM call failed on attempt {attempt}/{max_retries}: {e}")
            logger.debug(f"{log_prefix}Full traceback:\n{tb}")

            if agent_name:
                progress.update_status(agent_name, None, f"Retry {attempt}/{max_retries}")

            if attempt == max_retries:
                logger.error(f"{log_prefix}All retry attempts exhausted")
                return default_factory() if default_factory else create_default_response(pydantic_model)
            
            # Add delay before retry
            if attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"{log_prefix}Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

    return create_default_response(pydantic_model)

def get_embedding_model(model_name: str, model_provider: str) -> Optional[Any]:
    model_info = get_model_info(model_name)
    if not model_info:
        logger.error(f"Model info not found for {model_name}")
        return None
    try:
        if model_provider == "OpenAI":
            # Lazy-load OpenAI to avoid unnecessary API validation at startup
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=model_name)
        elif model_provider == "Gemini":
            return GoogleGenerativeAIEmbeddings(model=f"models/{os.getenv('EMB_MODEL', 'gemini-embedding-001')}")
        elif model_provider == "Ollama":
            return OllamaEmbeddings(model=model_name)
        else:
            logger.error(f"Embedding not supported for provider: {model_provider}")
            return None
    except Exception as e:
        logger.error(f"Error initializing embedding model: {e}")
        return None

# --- Other Utilities ---

def clean_documents(docs, min_content_length: int = 20, verbose: bool = False) -> Tuple[List, Dict]:
    stats = {"total": len(docs), "short": 0, "duplicate": 0, "valid": 0, "removed_docs": []}
    cleaned_docs = []
    content_map = {}

    for i, doc in enumerate(docs):
        doc.metadata["page"] = i + 1
        content = str(doc.page_content)
        content = content.replace(chr(65533), "ti")
        content = re.sub(r"\s*\n\s*", " ", content)
        content = re.sub(r"\n{2,}", "\n", content)
        content = re.sub(r"[ \t]{2,}", " ", content).strip()
        doc.page_content = content

        if len(content) < min_content_length:
            stats["short"] += 1
            stats["removed_docs"].append({"reason": "short", "page": i + 1, "metadata": doc.metadata.copy()})
            continue

        if content in content_map:
            stats["duplicate"] += 1
            stats["removed_docs"].append({"reason": "duplicate", "original_page": content_map[content] + 1, "page": i + 1, "metadata": doc.metadata.copy()})
            continue

        content_map[content] = i
        cleaned_docs.append(doc)

    stats["valid"] = len(cleaned_docs)
    return cleaned_docs, stats

console = Console()

class AgentProgress:
    def __init__(self):
        self.agent_status: Dict[str, Dict[str, str]] = {}
        self.table = Table(show_header=False, box=None, padding=(0, 1))
        self.live = Live(self.table, console=console, refresh_per_second=4)
        self.started = False

    def start(self):
        if not self.started:
            self.live.start()
            self.started = True

    def stop(self):
        if self.started:
            self.live.stop()
            self.started = False

    def update_status(self, agent_name: str, ticker: Optional[str] = None, status: str = ""):
        if agent_name not in self.agent_status:
            self.agent_status[agent_name] = {"status": "", "ticker": None}
        if ticker:
            self.agent_status[agent_name]["ticker"] = ticker
        if status:
            self.agent_status[agent_name]["status"] = status
        self._refresh_display()

    def _refresh_display(self):
        self.table.columns.clear()
        self.table.add_column(width=100)
        
        # Sort logic
        def sort_key(item):
            agent_name = item[0]
            if "risk_management" in agent_name: return (2, agent_name)
            elif "portfolio_management" in agent_name: return (3, agent_name)
            else: return (1, agent_name)

        for agent_name, info in sorted(self.agent_status.items(), key=sort_key):
            status = info["status"]
            ticker = info["ticker"]
            if status.lower() == "done":
                style = RichStyle(color="green", bold=True)
                symbol = "✓"
            elif status.lower() == "error":
                style = RichStyle(color="red", bold=True)
                symbol = "✗"
            else:
                style = RichStyle(color="yellow")
                symbol = "⋯"
            
            agent_display = agent_name.replace("_agent", "").replace("_", " ").title()
            status_text = Text()
            status_text.append(f"{symbol} ", style=style)
            status_text.append(f"{agent_display:<20}", style=RichStyle(bold=True))
            if ticker:
                status_text.append(f"[{ticker}] ", style=RichStyle(color="cyan"))
            status_text.append(status, style=style)
            self.table.add_row(status_text)

progress = AgentProgress()

# --- Citation Management ---

class CitationManager:
    """
    Central utility for handling citations across the TrueNorth RAG pipeline.
    """

    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ""
        return text.replace("\ufeff", "").strip()

    @staticmethod
    def _clean_snippet(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"#+\s*", "", text)
        text = re.sub(r"(\*\*|__|\*|_)", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        return text.strip()

    @staticmethod
    def normalize_document(doc: Document) -> CitationSource:
        metadata = doc.metadata or {}
        page_content = doc.page_content or ""

        def get_meta(key, default=""):
            val = metadata.get(key)
            if val is None: return default
            return str(val).replace("\ufeff", "").strip() or default

        url_meta = get_meta("url")
        source_meta = get_meta("source")
        file_path_meta = get_meta("file_path")

        is_web = False
        if url_meta:
            is_web = True
        elif source_meta and source_meta.startswith("http"):
            is_web = True

        author = get_meta("author", "Unknown Author")
        if author.startswith("{") and author.endswith("}"):
            author = author[1:-1]
        
        title = get_meta("title", "Unknown Title")
        year = get_meta("year")
        if not year:
            cdate = get_meta("creationdate", "")
            if len(cdate) >= 4: year = cdate[:4]
            else: year = "n.d."
        
        page = get_meta("page") or get_meta("page_number") or get_meta("page_num") or ""

        if is_web:
            final_url = url_meta or source_meta
            if title == "Unknown Title": title = final_url
            if author == "Unknown Author":
                try:
                    domain = urlparse(final_url).netloc
                    author = domain.replace("www.", "")
                except: author = "Web Source"
            filename = "web_source"
            doc_type = "web"
        else:
            file_path = file_path_meta or source_meta
            filename = os.path.basename(file_path) if file_path else "unknown.pdf"
            page_val = page if page else "1"
            encoded_filename = quote(filename)
            final_url = f"{FRONTEND_URL}/document?file={encoded_filename}&page={page_val}"
            doc_type = "pdf"

        extracted_quote = get_meta("extracted_quote", "")
        clean_content = CitationManager._clean_snippet(page_content)
        if extracted_quote:
            snippet = CitationManager._clean_snippet(extracted_quote)
        else:
            snippet = clean_content[:200]
            if len(clean_content) > 200: snippet += "..."

        return CitationSource(
            source_id=0, type=doc_type, author=author, title=title, year=year, 
            page=page, url=final_url, filename=filename, content_snippet=snippet, 
            full_content=page_content, quote=extracted_quote
        )

    @staticmethod
    def process_documents(state: ChatState) -> ChatState:
        documents = state.documents
        if not documents: return state

        registry = state.citation_registry
        current_max_id = max(registry.keys()) if registry else 0
        next_id = current_max_id + 1

        existing_lookup = {src.url: src.source_id for src in registry.values()}
        updated_docs = []

        for doc in documents:
            if isinstance(doc, dict):
                doc_obj = Document(page_content=doc.get("page_content", ""), metadata=doc.get("metadata", {}))
            else:
                doc_obj = doc

            source = CitationManager.normalize_document(doc_obj)
            unique_key = source.url

            if unique_key in existing_lookup:
                source.source_id = existing_lookup[unique_key]
            else:
                source.source_id = next_id
                existing_lookup[unique_key] = next_id
                registry[next_id] = source
                next_id += 1

            doc_obj.metadata["source_id"] = source.source_id
            doc_obj.metadata["citation_num"] = source.source_id 
            updated_docs.append(doc_obj)

        state.documents = updated_docs
        state.citation_registry = registry
        return state

    @staticmethod
    def get_context_string(state: ChatState) -> str:
        registry = state.citation_registry
        if not registry: return "No sources available."
        
        context_parts = []
        sorted_sources = sorted(registry.values(), key=lambda x: x.source_id)
        
        for src in sorted_sources:
            header = f"Source [{src.source_id}]:"
            meta_str = f"{src.author} ({src.year}). {src.title}."
            if src.page: meta_str += f" p.{src.page}"
            content = src.full_content.strip()
            block = f"{header}\nMetadata: {meta_str}\nContent: {content}\n"
            context_parts.append(block)

        return "\n---\n".join(context_parts)

    @staticmethod
    def resolve_citations(state: ChatState) -> Tuple[List[Dict[str, Any]], str]:
        registry = state.citation_registry
        llm_citations = state.generated_citations
        response_text = state.generation or ""
        final_citations = []

        used_ids = set()
        for cited in llm_citations: used_ids.add(cited.source_id)
        
        text_ids = set(int(x) for x in re.findall(r"\[(\d+)\]", response_text))
        used_ids.update(text_ids)

        # Check for invalid citation IDs (not in registry)
        invalid_ids = [id for id in used_ids if id not in registry]
        if invalid_ids:
            logger.warning(f"⚠️  Invalid citation IDs found in text: {sorted(invalid_ids)}")
            logger.warning(f"Available citation IDs in registry: {sorted(registry.keys())}")
        
        # Only include valid IDs in the mapping
        valid_used_ids = [id for id in used_ids if id in registry]
        sorted_used_ids = sorted(valid_used_ids)
        id_map = {old_id: new_id for new_id, old_id in enumerate(sorted_used_ids, start=1)}

        def replace_match(match):
            old_id = int(match.group(1))
            if old_id in id_map:
                return f"[{id_map[old_id]}]"
            # Remove invalid citations from text
            logger.debug(f"Removing invalid citation [{old_id}] from text")
            return ""

        renumbered_text = re.sub(r"\[(\d+)\]", replace_match, response_text)

        quotes_map = {}
        for cited in llm_citations:
            if cited.source_id not in quotes_map: quotes_map[cited.source_id] = cited.quote

        for old_id in sorted_used_ids:
            new_id = id_map[old_id]
            source = registry[old_id]
            quote = quotes_map.get(old_id, "")
            snippet = CitationManager._clean_snippet(quote)
            if not snippet or len(snippet) < 10: snippet = source.content_snippet

            cit = SimpleCitation(
                id=new_id, author=source.author, title=source.title, year=source.year, 
                page=source.page, snippet=snippet, url=source.url, filename=source.filename
            )
            final_citations.append(cit)

        return [c.model_dump() for c in final_citations], renumbered_text

# --- Helper Functions ---

def summarize_history_if_long(state: ChatState, model_name: str, model_provider: str, call_llm_fn):
    if len(state.messages) <= 6:
        return state 
    
    history_text = "\n".join(
        [f"User: {m.content}" if isinstance(m, HumanMessage) else f"Agent: {m.content}"
         for m in state.messages]
    )
    prompt = [
        SystemMessage("You are a helpful assistant. Summarize the conversation below into a concise summary that retains all essential points."),
        HumanMessage(content=history_text)
    ]
    summary_response = call_llm_fn(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=None,
        agent_name="history_summarizer"
    )
    summary_content = str(summary_response.content)
    last_messages = state.messages[-6:]
    state.messages = [AIMessage(content=f"[Summary of previous conversation]: {summary_content}")] + last_messages
    return state

def build_messages_for_llm(state: ChatState, current_question: str):
    return state.messages + [HumanMessage(content=current_question)]

def show_agent_reasoning(output, agent_name):
    print(f"\n{'=' * 10} {agent_name.center(28)} {'=' * 10}")
    def convert_to_serializable(obj):
        if hasattr(obj, "to_dict"): return obj.to_dict()
        elif hasattr(obj, "__dict__"): return obj.__dict__
        elif isinstance(obj, (int, float, bool, str)): return obj
        elif isinstance(obj, (list, tuple)): return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict): return {key: convert_to_serializable(value) for key, value in obj.items()}
        else: return str(obj)

    if isinstance(output, (dict, list)):
        print(json.dumps(convert_to_serializable(output), indent=4))
    else:
        try:
            print(json.dumps(json.loads(str(output)), indent=4))
        except:
            print(output)
    print("=" * 48)

# --- Agents & Nodes ---

class ReferenceSummary(BaseModel):
    summary: str
    key_quote: str

create_helpful_table_prompt_template = PromptTemplate.from_template(
    """
Today is {current_datetime}.

You are an assistant helping summarize articles into structured reference entries that are most useful to a user's question.

---

**Context Documents**:
{context}

---

**User Question**:
{question}

---

1. Extract the main idea or theme of the document as a short summary.
2. Select a key quote that represents the tone or message of the piece.
3. Return a JSON object with two keys: summary, and key_quote.
 
---

Your task:
Only return a single valid JSON object, nothing else.
                                                                    

"""
)

# Query Rewriter
query_rewriter_prompt_template = PromptTemplate.from_template(
    """
You are a query optimization expert tasked with rewriting questions to improve vector database retrieval accuracy.

---

**Context**:
- Original Question: {question}
- Previous Answer (incomplete or unhelpful): {generation}

**Vectorstore Summary**:
{vectorstore_content_summary}
                                                              
Note: The summary provides context about what's in the database but should not be treated as exhaustive.                                                       

---

**Your Task**:
Analyze the original question and the failed answer to identify:
1. What key information the original question was missing
2. Any ambiguities or unclear phrasing
3. Missing context or specialized terminology that should be included
4. Better keywords, phrasing, or terms to improve retrieval

---
                                                              
**Output Format**:
Return a JSON object with keys: "rewritten_question" and "explanation".
- "rewritten_question":  A refined version of the user's question optimized for vector search
- "explanation": A short explanation of how the rewrite improves coverage or clarity
"""
)

class QueryRewriteOutput(BaseModel):
    rewritten_question: str
    explanation: str

def rewrite_query(state: ChatState) -> ChatState:
    logger.info("=" * 60)
    logger.info("AGENT: Query Rewriter - START")
    logger.info("=" * 60)
    original_question = state.original_question if state.original_question else state.question
    generation = state.generation
    
    logger.info(f"Original question: {original_question}")
    logger.info(f"Current generation: {generation[:100]}..." if len(generation) > 100 else f"Current generation: {generation}")
    
    prompt = query_rewriter_prompt_template.format(
        question=original_question, generation=generation, 
        vectorstore_content_summary=vectorstore_content_summary
    )
    response = call_llm(
        prompt=[prompt],
        model_name=state.metadata.get("model_name"),
        model_provider=state.metadata.get("model_provider"),
        pydantic_model=QueryRewriteOutput,
        agent_name="query_rewriter",
        max_retries=1,
        verbose=True,
    )
    state.original_question = state.question
    state.question = response.rewritten_question
    
    logger.info(f"Rewritten question: {response.rewritten_question}")
    logger.info(f"Explanation: {response.explanation}")
    logger.info("AGENT: Query Rewriter - END")
    logger.info("=" * 60)
    return state

# Intent Description Agent
class IntentAnalysis(BaseModel):
    user_intent: str = Field(description="What the user truly wants to accomplish")
    intent_type: Literal["informational", "procedural", "emotional", "exploratory"]
    clarity_level: Literal["clear", "ambiguous", "compound"]
    key_topics: List[str] = Field(description="Main topics/themes in the question")
    refined_question: Optional[str] = Field(default=None, description="Clearer version if original is ambiguous")

intent_description_prompt = PromptTemplate.from_template(
    """You are an expert at understanding user intent and extracting what users truly need.

**Conversation History:**
{chat_history}

**Current Question:**
{question}

**Your Task:**
Analyze the question to understand what the user truly wants. Extract:

1. **user_intent**: The actual goal/need behind the question (be specific and action-oriented)
2. **intent_type**: Classify as one of:
   - "informational": User wants to learn/understand something
   - "procedural": User wants step-by-step guidance or how-to instructions
   - "emotional": User needs support, validation, or empathy
   - "exploratory": User is broadly exploring a topic to understand the landscape
   
3. **clarity_level**:
   - "clear": Question is specific and unambiguous
   - "ambiguous": Question is vague or could mean multiple things
   - "compound": Question contains multiple sub-questions
   
4. **key_topics**: List 2-4 main topics/themes (e.g., ["mentorship", "STEM careers"])

5. **refined_question**: If clarity_level is "ambiguous" or "compound", provide a clearer version

Return structured JSON.
""")

def intent_description_agent(state: ChatState) -> ChatState:
    logger.info("=" * 60)
    logger.info("AGENT: Intent Description - START")
    logger.info("=" * 60)
    
    question = state.question
    logger.info(f"Analyzing question: {question}")
    
    # Build minimal chat history context
    hist = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Agent'}: {m.content[:100]}..."
        for m in state.messages[-4:]  # Last 2 exchanges
    ) if state.messages else "(First turn)"
    
    prompt = intent_description_prompt.format(
        question=question,
        chat_history=hist
    )
    
    try:
        response = call_llm(
            prompt=[prompt],
            model_name=state.metadata.get("model_name"),
            model_provider=state.metadata.get("model_provider"),
            pydantic_model=IntentAnalysis,
            agent_name="intent_description"
        )
        
        state.user_intent = response.user_intent
        state.intent_metadata = {
            "intent_type": response.intent_type,
            "clarity_level": response.clarity_level,
            "key_topics": response.key_topics
        }
        
        # If question is ambiguous/compound and we have a refined version, use it
        if response.refined_question and response.clarity_level in ["ambiguous", "compound"]:
            logger.info(f"Refining ambiguous question")
            logger.info(f"Original: {question}")
            logger.info(f"Refined: {response.refined_question}")
            state.question = response.refined_question
        
        logger.info(f"✓ Extracted Intent: {response.user_intent}")
        logger.info(f"  Type: {response.intent_type}")
        logger.info(f"  Clarity: {response.clarity_level}")
        logger.info(f"  Topics: {response.key_topics}")
        
    except Exception as e:
        logger.warning(f"Intent analysis failed: {e}")
        # Set defaults if analysis fails
        state.user_intent = question
        state.intent_metadata = {
            "intent_type": "informational",
            "clarity_level": "clear",
            "key_topics": []
        }
    
    logger.info("AGENT: Intent Description - END")
    logger.info("=" * 60)
    return state

# Relevance Grader
class DocumentGrade(BaseModel):
    relevant: bool
    quote: Optional[str]

class BatchRelevanceGrade(BaseModel):
    grades: List[DocumentGrade]

batch_relevance_grader_prompt = PromptTemplate.from_template(
    """ou are a relevance grader evaluating a list of retrieved documents to see if they are helpful in answering a user question.

---

**User Question**:
{question}

**Documents**:
{formatted_documents}

---

**Instructions**:
- Evaluate each document in order.
- Return a JSON object with a single key `grades` which is a list of objects.
- Each object must have:
  - `relevant`: boolean (true/false)
  - `quote`: string (The exact sentence or paragraph from the text that is relevant. If not relevant, empty string.)
- The list order MUST match the document order (Doc 1 -> index 0)."""
)

async def check_relevance(state: ChatState) -> ChatState:
    logger.info("=" * 60)
    logger.info("AGENT: Relevance Grader - START")
    logger.info("=" * 60)
    
    if not state.documents:
        logger.warning("No documents to grade - marking as fail")
        state.metadata["relevance_score"] = "fail"
        logger.info("AGENT: Relevance Grader - END")
        logger.info("=" * 60)
        return state

    question = state.question
    logger.info(f"Question: {question}")
    logger.info(f"Documents to grade: {len(state.documents)}")
    
    # Use intent context for better relevance grading
    intent_type = state.intent_metadata.get("intent_type") if state.intent_metadata else None
    user_intent = state.user_intent
    
    intent_guidance = ""
    if intent_type and user_intent:
        intent_guidance = f"\n**User's True Intent**: {user_intent}\n**Intent Type**: {intent_type}\n"
        if intent_type == "procedural":
            intent_guidance += "Prioritize documents with actionable steps, how-to guides, and practical advice.\n"
        elif intent_type == "emotional":
            intent_guidance += "Prioritize documents with supportive language, validation, and empathetic guidance.\n"
        elif intent_type == "informational":
            intent_guidance += "Prioritize comprehensive, well-researched information with citations.\n"
        elif intent_type == "exploratory":
            intent_guidance += "Accept broader, diverse sources that help explore the topic landscape.\n"
        logger.info(f"Using intent-aware grading: {intent_type}")
    
    formatted_docs = []
    for i, d in enumerate(state.documents, 1):
        content = d.page_content.replace('\n', ' ').strip()[:800]
        formatted_docs.append(f"[Doc {i}] {content}...")

    formatted_docs_str = "\n\n".join(formatted_docs)
    
    # Enhanced prompt with intent context
    prompt_text = f"""{batch_relevance_grader_prompt.format(question=question, formatted_documents=formatted_docs_str)}
{intent_guidance}
Grade documents based on relevance to the TRUE INTENT, not just keyword matching."""

    try:
        result = await call_llm(
            prompt=[prompt_text],
            model_name=os.getenv("MODEL_NAME_LITE", "models/gemini-3.1-flash-lite-preview"),
            model_provider=os.getenv("MODEL_PROVIDER", "Google"),
            pydantic_model=BatchRelevanceGrade,
            agent_name="relevance_grader",
            max_retries=2
        )
        grades_list = result.grades
        logger.info(f"Successfully graded documents: {len(grades_list)} grades received")
    except Exception as e:
        logger.warning(f"Grading failed, defaulting all to relevant: {e}")
        grades_list = [DocumentGrade(relevant=True, quote="") for _ in state.documents]

    if len(grades_list) < len(state.documents):
        logger.warning(f"Grade list shorter than document list, padding with True grades")
        grades_list.extend([DocumentGrade(relevant=True, quote="") for _ in range(len(state.documents) - len(grades_list))])

    filtered_documents = []
    relevant_count = 0
    for doc, grade in zip(state.documents, grades_list):
        if grade.relevant:
            relevant_count += 1
            if grade.quote: 
                doc.metadata["extracted_quote"] = grade.quote
                logger.debug(f"Doc {relevant_count} - relevant with quote: {grade.quote[:50]}...")
            filtered_documents.append(doc)
    
    logger.info(f"Filtered documents: {len(filtered_documents)} relevant out of {len(state.documents)} total")
    state.documents = filtered_documents
    relevance_score = "pass" if len(filtered_documents) > 0 else "fail"
    state.metadata["relevance_score"] = relevance_score
    logger.info(f"Relevance score: {relevance_score}")
    
    state = CitationManager.process_documents(state)
    logger.info("AGENT: Relevance Grader - END")
    logger.info("=" * 60)
    return state

# Web Searcher
tavily_api_key = os.getenv("TAVILY_API_KEY")
web_search_tool = TavilySearchResults(max_results=5, tavily_api_key=tavily_api_key)

def validateURL(url: str) -> bool:
    try:
        r = requests.head(url, allow_redirects=True, timeout=5)
        return r.status_code < 400
    except: return False

def search_web(state: ChatState) -> ChatState:
    logger.info("=" * 60)
    logger.info("AGENT: Web Searcher - START")
    logger.info("=" * 60)
    logger.info(f"Question: {state.question}")
    logger.info(f"Documents in state before search: {len(state.documents)}")
    
    # Optimize search query based on intent
    intent_type = state.intent_metadata.get("intent_type") if state.intent_metadata else None
    key_topics = state.intent_metadata.get("key_topics", []) if state.intent_metadata else []
    
    search_query = state.question
    if intent_type and key_topics:
        logger.info(f"Intent-aware search: type={intent_type}, topics={key_topics}")
        if intent_type == "procedural":
            search_query = f"how to {state.question} step by step guide"
        elif intent_type == "informational":
            search_query = f"{state.question} comprehensive research"
        logger.info(f"Enhanced search query: {search_query}")
    
    try:
        logger.info("Invoking Tavily search tool...")
        web_results = web_search_tool.invoke(search_query)
        logger.info(f"Raw web results received: {len(web_results)}")
        
        documents = [Document(metadata={"url": d["url"], "title": d["title"]}, page_content=d["content"]) if isinstance(d, dict) else d for d in web_results]
        logger.info(f"Converted to {len(documents)} documents")
        
        documents, clean_stats = clean_documents(documents)
        logger.info(f"After cleaning: {len(documents)} documents (removed {clean_stats['short']} short, {clean_stats['duplicate']} duplicates)")
        
        initial_count = len(documents)
        documents = [d for d in documents if validateURL(d.metadata.get("url"))]
        logger.info(f"After URL validation: {len(documents)} valid documents ({initial_count - len(documents)} invalid URLs removed)")
        
        state.documents.extend(documents)
        logger.info(f"Total documents in state after web search: {len(state.documents)}")
        
        # Log some sample titles
        if documents:
            sample_titles = [d.metadata.get("title", "No title")[:50] for d in documents[:3]]
            logger.info(f"Sample titles: {sample_titles}")
        
        state = CitationManager.process_documents(state)
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    logger.info("AGENT: Web Searcher - END")
    logger.info("=" * 60)
    return state

# Document Retriever
multi_query_prompt = PromptTemplate.from_template(
    """You are an AI assistant helping improve document retrieval in a vector-based search system.

---
                                                             
**Context about the database**
The vectorstore contains the following content:
{vectorstore_content_summary}

Your goal is to help retrieve **more relevant documents** by rewriting a user's question from multiple angles.
This helps compensate for the limitations of semantic similarity in vector search.

---

**Instructions**:
Given the original question and the content summary above:
1. Return the **original user question** first.
2. Then generate {num_queries} **alternative versions** of the same question.
    - Rephrase using different word choices, structure, or focus.
    - Use synonyms or shift emphasis slightly, but keep the original meaning.
    - Make sure all rewrites are topically relevant to the database content.

Format requirements:
- Do **not** include bullet points or numbers.
- Each version should appear on a **separate newline**.
- Return **exactly {num_queries} + 1 total questions** (1 original + {num_queries} new ones).  

---                                              

**Original user question**: {question}"""
)

def reciprocal_rank_fusion(results, k=60):
    fused_scores = {}
    for docs in results:
        for i, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores: fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (i + 1 + k)
    
    reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [loads(doc_str) for doc_str, score in reranked]

def retrieve_documents(state: ChatState) -> ChatState:
    logger.info("=" * 60)
    logger.info("AGENT: Document Retriever - START")
    logger.info("=" * 60)
    logger.info(f"Question: {state.question}")
    logger.info(f"Documents in state before retrieval: {len(state.documents)}")
    
    # Leverage intent metadata for better retrieval
    intent_type = state.intent_metadata.get("intent_type") if state.intent_metadata else None
    clarity_level = state.intent_metadata.get("clarity_level") if state.intent_metadata else "clear"
    key_topics = state.intent_metadata.get("key_topics", []) if state.intent_metadata else []
    
    # Adjust retrieval parameters based on intent
    num_queries = {
        "clear": 2,       # Less variation needed
        "ambiguous": 4,   # More angles needed
        "compound": 4     # Multiple sub-intents
    }.get(clarity_level, 3)
    
    if intent_type and key_topics:
        logger.info(f"Intent-aware retrieval: type={intent_type}, clarity={clarity_level}, topics={key_topics}")
        logger.info(f"Using {num_queries} query variations")
    
    embedding_model = get_embedding_model(os.getenv("MODEL_NAME_SEARCH", "models/deep-research-pro-preview-12-2025"), "Gemini")
    
    if not os.path.exists(VECTOR_STORE_PATH):
        logger.error(f"Vector store not found at: {VECTOR_STORE_PATH}")
        logger.info("AGENT: Document Retriever - END")
        logger.info("=" * 60)
        return state

    try:
        logger.info("Loading FAISS vectorstore...")
        vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
        logger.info("Vectorstore loaded successfully")
        
        llm = lambda p: call_llm(p, os.getenv("MODEL_NAME_SEARCH", "models/deep-research-pro-preview-12-2025"), "Gemini", None)
        
        logger.info("Generating multi-query variations...")
        multi_query_generator = multi_query_prompt | llm | (lambda x: [line.strip() for line in str(x.content).split("\n") if line.strip()])
        retrieval_chain = multi_query_generator | vectorstore.as_retriever(search_type="mmr").map() | reciprocal_rank_fusion
        
        results = retrieval_chain.invoke({"question": state.question, "num_queries": num_queries, "vectorstore_content_summary": vectorstore_content_summary})
        logger.info(f"Retrieved {len(results)} documents from vectorstore")
        
        formatted = [Document(metadata={k:v for k,v in d.metadata.items() if k!="rrf_score"}, page_content=d.page_content) for d in results]
        logger.info(f"Formatted {len(formatted)} documents after RRF re-ranking")
        
        # Log sample sources
        if formatted:
            sample_sources = [d.metadata.get("source", "Unknown")[:50] for d in formatted[:3]]
            logger.info(f"Sample sources: {sample_sources}")
        
        state.documents.extend(formatted)
        logger.info(f"Total documents in state after retrieval: {len(state.documents)}")
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    logger.info("AGENT: Document Retriever - END")
    logger.info("=" * 60)
    return state

# Chitter Chatter
chitterchatter_prompt_template = PromptTemplate.from_template(
    """
Today is {current_datetime}. 

**Your goals are as following:**
{goals_as_str}

You do not replace a therapist, legal counsel, or HR department, but you can provide emotional support, educational context, helpful language, and confidential documentation tools.

Only use the links and chat history as mentioned below to support your advice.

---

**Current Scope**:
{system_relevant_scope}

**Chat History**:
{chat_history}


You are TrueNorth. Your job is to respond conversationally while gently guiding the user toward meaningful, empowering, and relevant discussions 
based on the resources in the knowledge base.

---

**Response Guidelines**:

1. **Casual Chit-Chat**:
  - Respond warmly to greetings or casual exchanges.
  - Keep the tone encouraging and human-like.
  - Be an empathetic listener if the user opens up.

2. **Off-Topic Questions**:
  - Politely acknowledge the question.
  - Mention that it falls outside your current scope.
  - Redirect to a related topic such as mentorship, leadership challenges, scholarships, navigating bias, or career growth in STEM.
  - Avoid saying "I don't know" without offering supportive redirection.

3. **In-Scope but Unanswerable Questions**:
  - If the question fits the mission but lacks enough detail to answer confidently:
    - Acknowledge the gap without guessing.
    - Gently ask for clarification or guide the user to rephrase the question.

4. **Model and System Questions**:
  - If asked about your identity, operational details, or ethical/economic concerns (e.g., "who are you?", "how do you work?", "what is your energy usage?"), respond with humility and transparency.
  - Acknowledge the importance of such questions and frame your existence as a tool designed to assist with leadership and workplace wellbeing.
  - Don't provide extra information that is not related to the question, like your energy usage or Gemini capabilities, unless they are asked for.
  - Only when asked for specific questions about underlying models (like Gemini), reference the following information:
    - Article Title: "Introducing Gemini: our largest and most capable AI model", Link: https://blog.google/technology/ai/google-gemini-ai/ -- if asked about about Gemini and capabilities
  - Only when asked for specific questions about environmental impact, reference the following information:
    - Fact: "One query uses 5 drops of water to generate." Article Title: "How much energy does Google’s AI use? We did the math", Link: https://cloud.google.com/blog/products/infrastructure/measuring-the-environmental-impact-of-ai-inference/) -- about water usage
    - Fact: "One query uses the same amount of energy as watching TV for 9 seconds." Video Title: "Calculating our AI energy consumption - Google Sustainability Report", Link: https://youtu.be/aarDw3sooYE?si=I8FZOl7-1LMp85A9) -- link this video if they ask about energy usage.
  - Be optimistic and reassuring about the future of AI, but also realistic about the current state of AI.
  - Reassure the user of your purpose: to provide helpful, evidence-based guidance in a secure, private manner.

---

**Important**:
Never invent or guess answers using general world knowledge.  
Your role is to **maintain trust** and offer emotionally supportive, mission-aligned responses.

Always keep a short and concise manner of speaking.
"""
)

def chitter_chatter_agent(state: ChatState) -> ChatState:
    logger.info("=" * 60)
    logger.info("AGENT: Chitter Chatter - START")
    logger.info("=" * 60)
    logger.info(f"Question: {state.question}")
    logger.info(f"Chat history length: {len(state.messages)} messages")
    
    state = summarize_history_if_long(state, THINK_MODEL_NAME, state.metadata["model_provider"], call_llm)
    hist = "\n".join(f"User: {m.content}" if isinstance(m, HumanMessage) else f"Agent: {m.content}" for m in state.messages)
    logger.info(f"Chat history prepared, total chars: {len(hist)}")
    
    # Customize response based on intent type
    intent_type = state.intent_metadata.get("intent_type") if state.intent_metadata else None
    user_intent = state.user_intent
    
    additional_guidance = ""
    if intent_type == "emotional" and user_intent:
        additional_guidance = f"\n\n**Context**: The user has an emotional need: '{user_intent}'. Be extra warm, empathetic, and supportive. Validate their feelings and offer gentle guidance."
        logger.info("Tailoring response for emotional intent")
    
    prompt = [
        SystemMessage(chitterchatter_prompt_template.format(
            current_datetime=datetime.now().strftime("%Y-%m-%d"),
            goals_as_str=goals_as_str, system_relevant_scope=system_relevant_scope,
            chat_history=hist, question=state.question
        ) + additional_guidance),
        HumanMessage(content=state.question)
    ]
    
    logger.info("Generating conversational response...")
    response = call_llm(prompt, THINK_MODEL_NAME, state.metadata["model_provider"], None, agent_name="chitter_chatter")
    content = str(response.content) if hasattr(response, "content") else str(response)
    state.generation = content
    
    logger.info(f"Generated response length: {len(content)} chars")
    logger.info(f"Response preview: {content[:100]}..." if len(content) > 100 else f"Response: {content}")
    logger.info("AGENT: Chitter Chatter - END")
    logger.info("=" * 60)
    return state

# Query Router
query_router_prompt = PromptTemplate.from_template(
    """
You are an expert at analyzing user question and deciding which data source is best suited to answer them. You must choose **one** of the following options:

---
                                                            
Scope Definition:
Relevant questions are those related to **{system_relevant_scope}** and topics in **{vectorstore_content_summary}**
---

1. **Vectorstore**: Use this if the question can be answered by the **existing** content in the vectorstore. 
   The vectorstore contains information about **{vectorstore_content_summary}**.
                                                            
---                                                         
                                                                                                                          
2. **Websearch**: Use this if the question is **within scope** and meets **any** of the following criteria:
    - The answer **cannot** be found in the local vectorstore
    - The question requires **specific, up-to-date information** that may not be in the books (e.g. current events, recent research, specific statistics)
    - The question contains a link or reference to something outside the books that needs to be looked up
    - The question requires **more detailed or factual information** than what's available in the books (e.g. exact birth date, current events or references)
    - The topic is **time-sensitive** , **current**, or depends on recent events or updates    
                                                                                                                
---                                                         
                                                              
3. **Chitter-Chatter**: Use this if the question:
   - Is **not related** to the scope below, or
   - Is too **broad, casual, or off-topic** to be answered using vectorstore or websearch.
   
   Chitter-Chatter is a fallback agent that gives a friendly response and a follow-up to guide users back to relevant topics.

---                                                        

Your Task:
Analyze the user's question. Return a JSON object with one key `"signal"` and one value: `"Vectorstore"`, `"Websearch"`, or `"Chitter-Chatter"`.

"""
)
class QueryRouterSignal(BaseModel):
    signal: Literal["Websearch", "Vectorstore", "Chitter-Chatter"]

def query_router_agent(state: ChatState) -> ChatState:
    logger.info("=" * 60)
    logger.info("AGENT: Query Router - START")
    logger.info("=" * 60)
    question = state.messages[0].content
    logger.info(f"Routing question: {question}")
    logger.info(f"Model: {state.metadata.get('model_name')} ({state.metadata.get('model_provider')})")
    
    # Leverage intent metadata for smarter routing
    intent_type = state.intent_metadata.get("intent_type") if state.intent_metadata else None
    user_intent = state.user_intent
    
    if intent_type and user_intent:
        logger.info(f"Intent context: {intent_type} - '{user_intent}'")
        
        # Fast-track obvious emotional support requests
        if intent_type == "emotional":
            state.metadata["signal"] = "Chitter-Chatter"
            logger.info("🎯 Fast-tracked to Chitter-Chatter (emotional intent)")
            logger.info("AGENT: Query Router - END")
            logger.info("=" * 60)
            return state
    
    prompt = [
        SystemMessage(query_router_prompt.format(question=question, vectorstore_content_summary=vectorstore_content_summary, system_relevant_scope=system_relevant_scope)),
        HumanMessage(content=question)
    ]
    
    def default_signal(): return QueryRouterSignal(signal="Chitter-Chatter")
    
    logger.info("Analyzing question to determine routing...")
    response = call_llm(prompt, state.metadata["model_name"], state.metadata["model_provider"], QueryRouterSignal, default_factory=default_signal)
    state.metadata["signal"] = response.signal
    
    logger.info(f"⚡ ROUTING DECISION: {response.signal}")
    logger.info(f"Next step: {'Web Search' if response.signal == 'Websearch' else 'Document Retrieval' if response.signal == 'Vectorstore' else 'Conversational Response'}")
    logger.info("AGENT: Query Router - END")
    logger.info("=" * 60)
    return state

# Hallucination Checker
def check_hallucination(state: ChatState) -> ChatState:
    logger.info("=" * 60)
    logger.info("AGENT: Hallucination Checker - START")
    logger.info("=" * 60)
    
    gen = state.generation or ""
    logger.info(f"Checking generation (length: {len(gen)} chars)")
    
    # Adjust triggers based on intent type
    intent_type = state.intent_metadata.get("intent_type") if state.intent_metadata else None
    
    if intent_type == "emotional":
        # More lenient for emotional support - allow supportive language
        triggers = [
            "as an AI", "not in the provided text",
            "based on general knowledge"
        ]
        logger.info("Using lenient hallucination check for emotional intent")
    elif intent_type == "procedural":
        # Stricter for actionable advice
        triggers = [
            "as an AI", "I'm not sure", "might work", "probably",
            "not in the provided text", "based on general knowledge"
        ]
        logger.info("Using strict hallucination check for procedural intent")
    else:
        # Default triggers
        triggers = [
            "as an AI", "I don't know", "not in the provided text",
            "based on general knowledge", "no information available"
        ]
    
    found_triggers = [t for t in triggers if t in gen.lower()]
    
    # Intent-specific validation
    if intent_type == "procedural" and "step" not in gen.lower() and len(gen) > 100:
        logger.warning("⚠️ Procedural intent but no steps detected in response")
        state.metadata["evaluator_reason"] = "incomplete_procedural"
    elif found_triggers:
        state.metadata["evaluator_reason"] = "hallucination"
        logger.warning(f"⚠️ HALLUCINATION DETECTED - Found triggers: {found_triggers}")
    else:
        state.metadata["evaluator_reason"] = "grounded"
        logger.info("✓ Response appears grounded in provided context")
    
    logger.info(f"Evaluator reason: {state.metadata['evaluator_reason']}")
    logger.info("AGENT: Hallucination Checker - END")
    logger.info("=" * 60)
    return state

# Answer Generator
class AnswerResponse(BaseModel):
    answer: str
    citations: Optional[List[CitedSource]] = []

answer_generator_prompt = PromptTemplate.from_template(
    """
Today is {current_datetime}.
                                                                
You are an assistant for question-answering tasks.

Here are your goals:
{goals_as_str}

You do not replace a therapist, legal counsel, or HR department, but you can provide emotional support, educational context, helpful language, and confidential documentation tools.

**Available Sources for Citation:**
{source_context}

IMPORTANT: Use ONLY the source IDs listed above when citing. When referencing information, use the format [ID] (e.g., [1], [2]) in the text. Do NOT invent or use citation numbers that are not explicitly provided in the sources above.

---
**Chat History**:
{chat_history}


**Background Knowledge**:
The available sources above provide the necessary background knowledge.

**User Question**:
{question}                                                       
---
                                                                
**Instructions**:
1. Base your answer primarily on the background knowledge provided above.
2. ONLY use citation IDs that appear in the "Available Sources for Citation" section above (e.g., if sources [1], [2], [3] are provided, only use those numbers).
3. If the answer is **not present** in the knowledge, say so explicitly.
4. Be **comprehensive**, **accurate**, and **focused** 
5. Give a short and concise answer to match the age demographic to connect with a more casual audience, (your audience range are 20-30 year old individuals)
6. Provide **concrete, actionable advice** that an individual can use to improve their situation, not an organizational solution.
7. Only answer questions relevant to STEM, workplace support, or academic guidance.
8. Return your response in the specified structured format (answer text only).
9. Limit your sources to 3
10. Keep your text under 3000 words
---
**Important**:
- Never invent or guess answers using general world knowledge.  
- Your role is to **maintain trust** and offer emotionally supportive, mission-aligned responses while maintaining a casual environment.
- Always keep a lighthearted but **thorough** manner of speaking while providing a helpful answer to the question.
"""
)

def answer_generator(state: ChatState) -> ChatState:
    logger.info("=" * 60)
    logger.info("AGENT: Answer Generator - START")
    logger.info("=" * 60)
    
    question = state.original_question or state.question
    logger.info(f"Question: {question}")
    logger.info(f"Documents available: {len(state.documents)}")
    logger.info(f"Chat history messages: {len(state.messages)}")
    
    # Leverage intent metadata for response customization
    intent_type = state.intent_metadata.get("intent_type") if state.intent_metadata else None
    user_intent = state.user_intent
    
    # Customize instructions based on intent type
    style_instructions = ""
    if intent_type:
        logger.info(f"Intent-aware generation: type={intent_type}, intent='{user_intent}'")
        
        if intent_type == "procedural":
            style_instructions = "\n**Style Guide for Procedural Intent**: Structure your answer with clear, numbered steps. Use action verbs. Make it immediately actionable with specific instructions.\n"
        elif intent_type == "emotional":
            style_instructions = "\n**Style Guide for Emotional Intent**: Lead with empathetic acknowledgment. Use warm, supportive language. Validate their feelings before offering guidance.\n"
        elif intent_type == "informational":
            style_instructions = "\n**Style Guide for Informational Intent**: Provide comprehensive overview with context. Use well-cited evidence. Aim for depth and accuracy.\n"
        elif intent_type == "exploratory":
            style_instructions = "\n**Style Guide for Exploratory Intent**: Offer broad overview of the topic landscape. Present multiple angles or approaches. Help them navigate the topic space.\n"
    
    context = CitationManager.get_context_string(state)
    logger.info(f"Context string length: {len(context)} chars")
    
    hist = "\n".join(f"{type(m).__name__}: {m.content}" for m in state.messages)
    logger.info(f"Conversation history:\n{hist if hist else '(empty)'}")
    
    # Include intent context in the question
    question_with_intent = question
    if user_intent:
        question_with_intent = f"[User Intent: {user_intent}]\n\n{question}"
    
    prompt = answer_generator_prompt.format(
        current_datetime=datetime.now(), goals_as_str=goals_as_str,
        source_context=context, chat_history=hist, question=question_with_intent
    ) + style_instructions
    
    model_name = THINK_MODEL_NAME
    logger.info(f"Generating answer with model: {model_name}")
    
    response = call_llm(prompt, model_name, os.getenv("MODEL_PROVIDER", "Google"), AnswerResponse)
    
    if isinstance(response, AnswerResponse):
        text = response.answer
        logger.info(f"Response type: AnswerResponse with {len(response.citations) if response.citations else 0} citations")
    else:
        text = str(response.content) if hasattr(response, "content") else str(response)
        logger.info("Response type: Raw text")
        
    state.generation = text
    state.generated_citations = [] # Citations handled by regex in resolve_citations
    
    logger.info(f"Generated answer length: {len(text)} chars")
    logger.info(f"Answer preview: {text[:150]}..." if len(text) > 150 else f"Answer: {text}")
    
    if state.current_user_message:
        state.add_user_message(state.current_user_message)
    state.add_agent_message(text)
    
    logger.info("AGENT: Answer Generator - END")
    logger.info("=" * 60)
    return state

# Evaluator
def evaluate_answer(state: ChatState) -> ChatState:
    logger.info("=" * 60)
    logger.info("AGENT: Evaluator - START")
    logger.info("=" * 60)
    
    state.current_try += 1
    reason = state.metadata.get("evaluator_reason")
    
    logger.info(f"Current attempt: {state.current_try}/{state.max_retries}")
    logger.info(f"Evaluator reason: {reason}")
    
    if state.current_try >= state.max_retries:
        state.metadata["evaluator_result"] = "max_retries"
        logger.warning(f"⚠️  MAX RETRIES REACHED - stopping at attempt {state.current_try}")
    elif reason == "hallucination":
        state.metadata["evaluator_result"] = "hallucinated"
        logger.warning("⚠️  HALLUCINATION DETECTED - will trigger query rewrite")
    else:
        state.metadata["evaluator_result"] = "pass"
        logger.info("✓ Answer evaluation PASSED")
    
    logger.info(f"Evaluator result: {state.metadata['evaluator_result']}")
    logger.info(f"Next step: {'Query Rewrite' if state.metadata['evaluator_result'] == 'hallucinated' else 'Chitter Chatter' if state.metadata['evaluator_result'] == 'max_retries' else 'Complete'}")
    logger.info("AGENT: Evaluator - END")
    logger.info("=" * 60)
    return state

# --- Graph Construction ---

def build_rag_graph():
    builder = StateGraph(ChatState)
    builder.add_node("intent_description", intent_description_agent)
    builder.add_node("web_searcher", search_web)
    builder.add_node("document_retriever", retrieve_documents)
    builder.add_node("chitter_chatter", chitter_chatter_agent)
    builder.add_node("query_rewriter", rewrite_query)
    builder.add_node("evaluate_answer", evaluate_answer)
    builder.add_node("answer_generator", answer_generator)
    builder.add_node("relevance_grader", check_relevance)
    builder.add_node("query_router", query_router_agent)
    builder.add_node("hallucination_checker", check_hallucination)

    builder.add_edge(START, "intent_description")
    builder.add_edge("intent_description", "query_router")
    builder.add_conditional_edges("query_router", lambda s: s.metadata.get("signal", "Chitter-Chatter"), 
        {"Websearch": "web_searcher", "Vectorstore": "document_retriever", "Chitter-Chatter": "chitter_chatter"})
    
    builder.add_edge("document_retriever", "relevance_grader")
    builder.add_conditional_edges("relevance_grader", lambda s: s.metadata.get("relevance_score", "fail"), 
        {"fail": "web_searcher", "pass": "answer_generator"})
    
    builder.add_edge("web_searcher", "answer_generator")
    builder.add_edge("answer_generator", "hallucination_checker")
    builder.add_edge("hallucination_checker", "evaluate_answer")
    builder.add_conditional_edges("evaluate_answer", lambda s: s.metadata.get("evaluator_result", "fail"),
        {"hallucinated": "query_rewriter", "max_retries": "chitter_chatter", "pass": END})
    
    builder.add_edge("query_rewriter", "document_retriever")
    builder.add_edge("chitter_chatter", END)
    
    return builder

# --- PDF API & Router ---

pdf_router = APIRouter(prefix="/pdf", tags=["pdf"])

def get_pdf_path(filename: str) -> Path:
    filename = os.path.basename(filename)
    pdf_path = BOOKS_DIR / filename
    if not pdf_path.exists(): raise HTTPException(404, "PDF not found")
    if not pdf_path.is_relative_to(BOOKS_DIR): raise HTTPException(403, "Invalid path")
    return pdf_path

def render_pdf_pages(filename: str, page_num: int, range_before=2, range_after=2, dpi=150, highlight_text=None) -> list:
    pdf_path = get_pdf_path(filename)
    doc = fitz.open(pdf_path)
    try:
        total_pages = len(doc)
        target_idx = page_num - 1
        start_idx = max(0, target_idx - range_before)
        end_idx = min(total_pages - 1, target_idx + range_after)
        pages_data = []

        for idx in range(start_idx, end_idx + 1):
            page = doc[idx]
            if idx == target_idx and highlight_text:
                clean_text = " ".join(highlight_text.split())
                instances = page.search_for(clean_text)
                if instances:
                    shape = page.new_shape()
                    for rect in instances: shape.draw_rect(rect)
                    shape.finish(color=None, fill=(0.9, 0.9, 0.98), fill_opacity=0.5)
                    shape.commit()
            
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
            pages_data.append({
                "page_num": idx + 1, "image_base64": img_b64, "is_target": (idx == target_idx),
                "width": pix.width, "height": pix.height
            })
        return pages_data
    finally:
        doc.close()

@pdf_router.get("/{filename}/pages")
async def get_pdf_pages(filename: str, page: int = Query(..., ge=1), range_before: int = 2, range_after: int = 2, dpi: int = 150, highlight: str = None):
    try:
        pages = render_pdf_pages(filename, page, range_before, range_after, dpi, highlight)
        return {"pages": pages, "target_page": page, "filename": filename}
    except Exception as e: raise HTTPException(500, str(e))

@pdf_router.get("/{filename}/metadata")
async def get_metadata(filename: str):
    try:
        doc = fitz.open(get_pdf_path(filename))
        return {
            "filename": filename, "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""), "page_count": len(doc)
        }
    except Exception as e: raise HTTPException(500, str(e))

# --- Main App ---

@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_postgres_storage()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(pdf_router)
app.include_router(pdf_router, prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryInput(BaseModel):
    snowflake: str = 'test'
    question: str = ''
    chat_history: List[str] = []
    conversation_id: Optional[str] = None
    anonymized_history_consent: bool = False

class QueryOutput(BaseModel):
    response: str
    citations: List[SimpleCitation] = []


class ChatPreferencesInput(BaseModel):
    snowflake: str
    anonymized_history_consent: bool = False


class ChatPreferencesOutput(BaseModel):
    snowflake: str
    anonymized_history_consent: bool


# --- Logs API Models ---

class ConversationMessage(BaseModel):
    """Individual message in a conversation"""
    id: int
    role: Literal['user', 'assistant']
    content: str
    created_at: str  # ISO8601 timestamp
    payload: Dict[str, Any] = Field(default_factory=dict)


class ConversationSummary(BaseModel):
    """Summary of a conversation for list view"""
    conversation_id: Optional[str] = None
    snowflake: str
    message_count: int
    first_message_at: str  # ISO8601
    last_message_at: str   # ISO8601
    anonymized_history_consent: bool
    preview: str  # First user message (truncated)


class ConversationDetail(BaseModel):
    """Full conversation with all messages"""
    conversation_id: Optional[str] = None
    snowflake: str
    anonymized_history_consent: bool
    messages: List[ConversationMessage]
    created_at: str
    updated_at: str


class PaginatedResponse(BaseModel):
    """Generic paginated response wrapper"""
    items: List[Any] = Field(default_factory=list)
    total: int
    limit: int
    offset: int
    has_next: bool
    has_prev: bool


class ConversationsPaginatedResponse(BaseModel):
    """Paginated response for conversations"""
    items: List[ConversationSummary] = Field(default_factory=list)
    total: int
    limit: int
    offset: int
    has_next: bool
    has_prev: bool


class PipelineStage(BaseModel):
    """Individual stage in the agent pipeline execution"""
    agent_name: str
    status: Literal['started', 'completed', 'error']
    start_time: str  # ISO8601
    end_time: Optional[str] = None  # ISO8601
    duration_ms: Optional[int] = None
    status_message: str


class PipelineExecution(BaseModel):
    """Agent pipeline execution record"""
    id: str  # Generated from log timestamp
    snowflake: str
    question: str
    start_time: str  # ISO8601
    end_time: Optional[str] = None  # ISO8601
    duration_ms: Optional[int] = None
    agent_sequence: List[str] = Field(default_factory=list)
    stages: List[PipelineStage] = Field(default_factory=list)
    status: Literal['success', 'error', 'running']
    error_message: Optional[str] = None


class PipelineExecutionsPaginatedResponse(BaseModel):
    """Paginated response for pipeline executions"""
    items: List[PipelineExecution] = Field(default_factory=list)
    total: int
    limit: int
    offset: int
    has_next: bool
    has_prev: bool


class LogsFilters(BaseModel):
    """Query parameters for filtering logs"""
    date_from: Optional[str] = None  # ISO8601 date
    date_to: Optional[str] = None    # ISO8601 date
    snowflake: Optional[str] = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    sort_by: str = Field(default="created_at")
    sort_order: Literal['asc', 'desc'] = Field(default='desc')
    consent_only: Optional[bool] = None  # Filter by anonymized_history_consent

AGENT_STATUS_MESSAGES = {
    "query_router": "Analyzing your question...",
    "web_searcher": "Searching the web...",
    "document_retriever": "Searching internal knowledge...",
    "chitter_chatter": "Thinking...",
    "query_rewriter": "Refining search...",
    "evaluate_answer": "Evaluating quality...",
    "answer_generator": "Writing response...",
    "relevance_grader": "Checking relevance...",
    "hallucination_checker": "Verifying facts...",
}

async def stream_workflow(
    question: str,
    snowflake: str,
    conversation_id: Optional[str] = None,
    chat_history: Optional[List[str]] = None,
    anonymized_history_consent: bool = False,
):
    logger.info("=" * 80)
    logger.info("🚀 WORKFLOW EXECUTION START")
    logger.info("=" * 80)
    snowflake = snowflake or "test"
    consent_for_turn = anonymized_history_consent
    logger.info(f"Question: {question}")
    logger.info(f"Snowflake: {snowflake}")
    logger.info(f"Anonymized history consent: {consent_for_turn}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    workflow = build_rag_graph()
    agent = workflow.compile()
    
    state = get_state(snowflake)

    previous_conversation_id = state.metadata.get("conversation_id")
    has_incoming_history = bool(chat_history)
    is_new_conversation = (
        (conversation_id is not None and conversation_id != previous_conversation_id)
        or (not has_incoming_history)
    )

    if is_new_conversation:
        logger.info(
            "Detected new conversation. Clearing persisted chat history for snowflake '%s' (prev_conversation_id=%s, new_conversation_id=%s, incoming_history_len=%s)",
            snowflake,
            previous_conversation_id,
            conversation_id,
            0 if chat_history is None else len(chat_history),
        )
        state.clear_conversation()
        state.documents.clear()
        state.citation_registry.clear()
        state.generated_citations.clear()
        state.current_try = 0

    state.metadata["conversation_id"] = conversation_id
    state.anonymized_history_consent = consent_for_turn
    state.question = question
    state.original_question = question
    state.current_user_message = question
    state.add_user_message(question, anonymized_history_consent=consent_for_turn)
    save_state(state)
    log_chat_history(
        snowflake=snowflake,
        role="user",
        content=question,
        conversation_id=conversation_id,
        anonymized_history_consent=consent_for_turn,
        payload={"event": "user_message"},
    )
    
    inputs = {
        "snowflake": snowflake,
        "question": question,
        "messages": state.messages,
        "anonymized_history_consent": consent_for_turn,
        "metadata": {"model_name": os.getenv("MODEL_NAME_LITE", "models/gemini-3.1-flash-lite-preview"), "model_provider": os.getenv("MODEL_PROVIDER", "Google")}
    }
    
    logger.info(f"Initial state - Messages: {len(state.messages)}, Model: {inputs['metadata']['model_name']}")

    final_state_dict = None
    agent_sequence = []
    start_time = time.time()
    
    try:
        async for event in agent.astream_events(inputs, version="v2"):
            kind = event["event"]
            event_name = event.get("name", "")
            
            if kind == "on_chain_start" and event_name in AGENT_STATUS_MESSAGES:
                logger.info(f"▶️  Agent started: {event_name}")
                agent_sequence.append(event_name)
                yield f"data: {json.dumps({'type': 'status', 'message': AGENT_STATUS_MESSAGES[event_name]})}\n\n"
            elif kind == "on_chain_end":
                if event_name in AGENT_STATUS_MESSAGES:
                    logger.info(f"✓ Agent completed: {event_name}")
                elif event_name == "LangGraph":
                    final_state_dict = event["data"].get("output")
                    logger.info("✓ LangGraph workflow completed")

        if not final_state_dict:
            logger.warning("No final state from stream, invoking synchronously...")
            final_state_dict = await agent.ainvoke(inputs)

        final_state = ChatState(**final_state_dict)
        citations, renumbered_text = CitationManager.resolve_citations(final_state)
        state.generation = renumbered_text
        state.current_user_message = None
        latest_consent = get_state(snowflake).anonymized_history_consent
        state.anonymized_history_consent = latest_consent
        state.add_agent_message(renumbered_text, anonymized_history_consent=consent_for_turn)
        save_state(state)
        log_chat_history(
            snowflake=snowflake,
            role="assistant",
            content=renumbered_text,
            conversation_id=conversation_id,
            anonymized_history_consent=consent_for_turn,
            payload={"event": "assistant_message", "citations": citations},
        )
    except Exception as e:
        logger.error(f"❌ Stream error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        state.current_user_message = None
        state.anonymized_history_consent = get_state(snowflake).anonymized_history_consent
        save_state(state)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        return
    
    duration = time.time() - start_time
    logger.info(f"📊 Workflow Summary:")
    logger.info(f"   Duration: {duration:.2f}s")
    logger.info(f"   Agent sequence: {' → '.join(agent_sequence)}")
    logger.info(f"   Final answer length: {len(renumbered_text)} chars")
    logger.info(f"   Citations: {len(citations)}")
    logger.info("=" * 80)
    logger.info("✅ WORKFLOW EXECUTION COMPLETE")
    logger.info("=" * 80)
    
    yield f"data: {json.dumps({'type': 'result', 'response': renumbered_text, 'citations': citations})}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/api/chat/preferences", response_model=ChatPreferencesOutput)
async def get_chat_preferences(snowflake: str = Query(..., min_length=1)):
    state = get_state(snowflake)
    return ChatPreferencesOutput(
        snowflake=state.snowflake,
        anonymized_history_consent=state.anonymized_history_consent,
    )


@app.post("/api/chat/preferences", response_model=ChatPreferencesOutput)
async def update_chat_preferences(input_data: ChatPreferencesInput):
    if not input_data.snowflake:
        raise HTTPException(status_code=400, detail="snowflake is required")

    state = get_state(input_data.snowflake)
    state.anonymized_history_consent = input_data.anonymized_history_consent
    save_state(state)
    return ChatPreferencesOutput(
        snowflake=state.snowflake,
        anonymized_history_consent=state.anonymized_history_consent,
    )

@app.post("/query", response_model=QueryOutput)
async def get_chat_response(input_data: QueryInput):
    # Non-streaming wrapper
    result_chunks = [
        chunk
        async for chunk in stream_workflow(
            input_data.question,
            input_data.snowflake,
            input_data.conversation_id,
            input_data.chat_history,
            input_data.anonymized_history_consent,
        )
    ]
    
    # Extract final result
    final_output = None
    for chunk in result_chunks:
        if '"type": "result"' in chunk:
            final_output = json.loads(chunk.replace("data: ", ""))
            break
            
    if final_output:
        return QueryOutput(response=final_output["response"], citations=[SimpleCitation(**c) for c in final_output["citations"]])
    return QueryOutput(response="Error generating response")

@app.post("/api/chat/stream")
async def stream_chat_response(request: Request):
    try:
        data = await request.json()
        input_data = QueryInput(**data)
        return StreamingResponse(
            stream_workflow(
                input_data.question,
                input_data.snowflake,
                input_data.conversation_id,
                input_data.chat_history,
                input_data.anonymized_history_consent,
            ),
            media_type="text/event-stream",
        )
    except Exception as e: return JSONResponse({"error": str(e)}, 500)



# --- Logs API Endpoints ---

@app.get("/api/logs/conversations", response_model=ConversationsPaginatedResponse)
async def get_conversations(
    date_from: Optional[str] = Query(None, description="ISO8601 date filter (from)"),
    date_to: Optional[str] = Query(None, description="ISO8601 date filter (to)"),
    snowflake: Optional[str] = Query(None, description="Filter by user snowflake"),
    consent_only: Optional[bool] = Query(None, description="Filter by consent status"),
    limit: int = Query(20, ge=1, le=100, description="Number of results per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: Literal['asc', 'desc'] = Query('desc', description="Sort order"),
):
    """
    Get paginated list of conversations with summary information.
    Groups chat_history_logs by conversation_id and returns metadata.
    """
    try:
        initialize_postgres_storage()
        
        with ChatLogSessionLocal() as session:
            # Add filters
            filter_conditions = []
            filter_params = {}
            
            if date_from:
                filter_conditions.append("created_at >= :date_from")
                filter_params['date_from'] = date_from
            if date_to:
                filter_conditions.append("created_at <= :date_to")
                filter_params['date_to'] = date_to
            if snowflake:
                filter_conditions.append("snowflake = :snowflake")
                filter_params['snowflake'] = snowflake
            if consent_only is not None:
                filter_conditions.append("anonymized_history_consent = :consent")
                filter_params['consent'] = consent_only
            
            # Build final query
            where_clause = " AND ".join(filter_conditions) if filter_conditions else "1=1"
            sort_direction = "DESC" if sort_order == 'desc' else "ASC"
            
            count_query = text(f"""
                SELECT COUNT(DISTINCT COALESCE(conversation_id, 'none_' || snowflake))
                FROM chat_history_logs
                WHERE {where_clause}
            """)
            
            main_query = text(f"""
                SELECT 
                    COALESCE(conversation_id, 'none_' || snowflake) as conv_id,
                    conversation_id,
                    snowflake,
                    COUNT(*) as message_count,
                    MIN(created_at) as first_message_at,
                    MAX(created_at) as last_message_at,
                    BOOL_OR(anonymized_history_consent) as anonymized_history_consent,
                    (
                        SELECT content 
                        FROM chat_history_logs logs
                        WHERE role = 'user'
                        AND logs.snowflake = main.snowflake
                        AND (logs.conversation_id = main.conversation_id OR (logs.conversation_id IS NULL AND main.conversation_id IS NULL))
                        ORDER BY created_at ASC
                        LIMIT 1
                    ) as first_user_message
                FROM chat_history_logs main
                WHERE {where_clause}
                GROUP BY conversation_id, snowflake
                ORDER BY MAX(created_at) {sort_direction}
                LIMIT :limit OFFSET :offset
            """)
            
            filter_params.update({'limit': limit, 'offset': offset})
            
            # Execute count query
            total = session.execute(count_query, filter_params).scalar() or 0
            
            # Execute main query
            result = session.execute(main_query, filter_params)
            rows = result.fetchall()
            
            # Build response
            items = []
            for row in rows:
                preview = (row[7] or '')[:100]  # Truncate preview
                if len(row[7] or '') > 100:
                    preview += '...'
                
                items.append(ConversationSummary(
                    conversation_id=row[1],
                    snowflake=row[2],
                    message_count=row[3],
                    first_message_at=row[4].isoformat() if row[4] else '',
                    last_message_at=row[5].isoformat() if row[5] else '',
                    anonymized_history_consent=row[6] or False,
                    preview=preview
                ))
            
            return ConversationsPaginatedResponse(
                items=items,
                total=total,
                limit=limit,
                offset=offset,
                has_next=offset + limit < total,
                has_prev=offset > 0
            )
    
    except Exception as e:
        logger.error(f"Error fetching conversations: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logs/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation_detail(conversation_id: str):
    """
    Get full conversation with all messages.
    """
    try:
        initialize_postgres_storage()
        
        with ChatLogSessionLocal() as session:
            # Query all messages for this conversation
            query = text("""
                SELECT id, role, content, created_at, payload, snowflake, anonymized_history_consent
                FROM chat_history_logs
                WHERE conversation_id = :conv_id OR (conversation_id IS NULL AND snowflake = :conv_id)
                ORDER BY created_at ASC
            """)
            result = session.execute(query, {"conv_id": conversation_id})
            rows = result.fetchall()
            
            if not rows:
                raise HTTPException(status_code=404, detail="Conversation not found")
            
            # Build messages list
            messages = []
            for row in rows:
                messages.append(ConversationMessage(
                    id=row[0],
                    role=row[1],
                    content=row[2],
                    created_at=row[3].isoformat() if row[3] else '',
                    payload=row[4] or {}
                ))
            
            # Use first message data for metadata
            first_row = rows[0]
            last_row = rows[-1]
            
            return ConversationDetail(
                conversation_id=conversation_id if conversation_id != first_row[5] else None,
                snowflake=first_row[5],
                anonymized_history_consent=first_row[6] or False,
                messages=messages,
                created_at=first_row[3].isoformat() if first_row[3] else '',
                updated_at=last_row[3].isoformat() if last_row[3] else ''
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching conversation detail: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logs/users/{snowflake}", response_model=ConversationsPaginatedResponse)
async def get_user_conversations(
    snowflake: str,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """
    Get all conversations for a specific user.
    """
    # Reuse the main conversations endpoint with snowflake filter
    return await get_conversations(
        snowflake=snowflake,
        limit=limit,
        offset=offset
    )


@app.get("/api/logs/export/{conversation_id}")
async def export_conversation(
    conversation_id: str,
    format: Literal['json', 'csv'] = Query('json', description="Export format")
):
    """
    Export conversation in JSON or CSV format.
    """
    try:
        # Get full conversation
        conversation = await get_conversation_detail(conversation_id)
        
        if format == 'json':
            return JSONResponse(
                content=conversation.dict(),
                headers={
                    "Content-Disposition": f'attachment; filename="conversation_{conversation_id}.json"'
                }
            )
        
        elif format == 'csv':
            import io
            import csv
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write headers
            writer.writerow(['ID', 'Role', 'Content', 'Created At', 'Snowflake', 'Conversation ID'])
            
            # Write data
            for msg in conversation.messages:
                writer.writerow([
                    msg.id,
                    msg.role,
                    msg.content,
                    msg.created_at,
                    conversation.snowflake,
                    conversation.conversation_id or ''
                ])
            
            csv_content = output.getvalue()
            
            from fastapi.responses import Response
            return Response(
                content=csv_content,
                media_type="text/csv",
                headers={
                    "Content-Disposition": f'attachment; filename="conversation_{conversation_id}.csv"'
                }
            )
    
    except Exception as e:
        logger.error(f"Error exporting conversation: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logs/pipeline/executions", response_model=PipelineExecutionsPaginatedResponse)
async def get_pipeline_executions(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """
    Get recent pipeline executions from log files.
    """
    try:
        # Parse log file
        all_executions = parse_pipeline_logs(limit=limit + offset)
        
        # Apply pagination
        paginated = all_executions[offset:offset + limit]
        total = len(all_executions)
        
        return PipelineExecutionsPaginatedResponse(
            items=paginated,
            total=total,
            limit=limit,
            offset=offset,
            has_next=offset + limit < total,
            has_prev=offset > 0
        )
    
    except Exception as e:
        logger.error(f"Error fetching pipeline executions: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logs/pipeline/executions/{execution_id}", response_model=PipelineExecution)
async def get_pipeline_execution_detail(execution_id: str):
    """
    Get detailed pipeline execution by ID (timestamp).
    """
    try:
        # Parse log file and find matching execution
        executions = parse_pipeline_logs(limit=500)
        
        for execution in executions:
            if execution.id == execution_id:
                return execution
        
        raise HTTPException(status_code=404, detail="Pipeline execution not found")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching pipeline execution detail: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    try:
        initialize_postgres_storage()
        return {
            "status": "ok",
            "postgres": {
                "host": POSTGRES_HOST,
                "port": POSTGRES_PORT,
                "chat_state_db": CHAT_STATE_DB_NAME,
                "chat_log_db": CHAT_LOG_DB_NAME,
            },
        }
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, 503)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
