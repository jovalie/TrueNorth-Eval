import json
import logging
from langchain_core.messages.base import BaseMessage
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import operator

# Merge utility (you can use it manually during runtime)
def merge_dicts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {**a, **b}


# --- Citation Data Models ---


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
    quote: str  # The exact text extracted by LLM

    # Content
    content_snippet: str  # Fallback content (~200 chars)
    full_content: str = Field(exclude=True)  # Full text for LLM context (excluded from serialization if needed)


class CitedSource(BaseModel):
    """Represents a citation used in the generated answer."""

    source_id: int
    quote: str = Field(description="The exact quote from the source that supports the claim.")


# Define agent state as a Pydantic model
class ChatState(BaseModel):
    snowflake: str

    question: str = ""
    original_question: Optional[str] = None
    generation: Optional[str] = None

    messages: list = Field(default_factory=list)
    current_user_message: str | None = None
    documents: List[Any] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    citation_registry: Dict[int, CitationSource] = Field(default_factory=dict)
    generated_citations: List[CitedSource] = Field(default_factory=list)

    max_retries: int = 2
    current_try: int = 0

    def add_user_message(self, content: str):
        self.messages.append(HumanMessage(content=content))

    def add_agent_message(self, content: str):
        self.messages.append(AIMessage(content=content))

    def clear_conversation(self):
        self.messages.clear()
    #discord snowflake
CHAT_STATES: Dict[str, ChatState] = {}
def get_state(user_snowflake: str) -> ChatState:
    if user_snowflake not in CHAT_STATES:
        CHAT_STATES[user_snowflake] = ChatState(
            snowflake=user_snowflake
        )
    return CHAT_STATES[user_snowflake]
def get_chat_history_text(state: ChatState) -> str:
    messages = []
    for msg in state.messages:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        messages.append(f"{role}: {msg.content}")
    return "\n".join(messages)

def summarize_history_if_long(state: ChatState, model_name: str, model_provider: str,call_llm_fn):
   #summarize if greater than 6 messages
    if len(state.messages) <= 6:
        return state 
    # Prepare history text for summarization
    history_text = "\n".join(
        [f"User: {m.content}" if isinstance(m, HumanMessage) else f"Agent: {m.content}"
         for m in state.messages]
    )
    # Summarization prompt
    prompt = [
        SystemMessage("You are a helpful assistant. Summarize the conversation below "
                      "into a concise summary that retains all essential points."),
        HumanMessage(content=history_text)
    ]

    # Call the same LLM to summarize
    summary_response = call_llm_fn(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=None,
        agent_name="history_summarizer"
    )

    # Replace old messages with the summary + last 6 messages for continuity
    summary_content = str(summary_response.content)
    last_messages = state.messages[-6:]
    state.messages = [AIMessage(content=f"[Summary of previous conversation]: {summary_content}")] + last_messages

    return state

def get_conversation(state: ChatState) -> List[BaseMessage]:
    return state.messages

class HCResult(BaseModel):
    binary_score: str
    explanation: Optional[str] = None


class AVResult(BaseModel):
    relevance_score: str
    explanation: Optional[str] = None

def build_messages_for_llm(state: ChatState, current_question: str):
    return state.messages + [HumanMessage(content=current_question)]
def show_agent_reasoning(output, agent_name):
    print(f"\n{'=' * 10} {agent_name.center(28)} {'=' * 10}")

    def convert_to_serializable(obj):
        if hasattr(obj, "to_dict"):  # Handle Pandas Series/DataFrame
            return obj.to_dict()
        elif hasattr(obj, "__dict__"):  # Handle custom objects
            return obj.__dict__
        elif isinstance(obj, (int, float, bool, str)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return str(obj)  # Fallback to string representation

    if isinstance(output, (dict, list)):
        # Convert the output to JSON-serializable format
        serializable_output = convert_to_serializable(output)
        print(json.dumps(serializable_output, indent=4))
    else:
        try:
            # Parse the string as JSON and pretty print it
            parsed_output = json.loads(str(output))
            print(json.dumps(parsed_output, indent=4))
        except json.JSONDecodeError:
            # Fallback to original string if not valid JSON
            print(output)

    print("=" * 48)
