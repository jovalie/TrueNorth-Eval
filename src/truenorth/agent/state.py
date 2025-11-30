import json
import logging
from langchain_core.messages.base import BaseMessage
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
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

    # Content
    content_snippet: str  # Fallback content (~200 chars)
    full_content: str = Field(exclude=True)  # Full text for LLM context (excluded from serialization if needed)


class CitedSource(BaseModel):
    """Output from the LLM Answer Generator."""

    source_id: int  # Matches CitationSource.source_id
    quote: str  # The exact text extracted by LLM


# Define agent state as a Pydantic model
class ChatState(BaseModel):
    question: str
    original_question: str = None
    generation: str = None

    # NOTE: messages and documents are stored redundantly — they also live inside metadata variable
    messages: List[BaseMessage] = Field(default_factory=list)
    documents: List[Any] = Field(default_factory=list)

    # Contains all information carried between agents
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # NEW: Explicit Citation Registry
    # Maps source_id (int) -> CitationSource
    citation_registry: Dict[int, CitationSource] = Field(default_factory=dict)

    # NEW: Structured Output from Answer Generator
    generated_citations: List[CitedSource] = Field(default_factory=list)

    max_retries: int = 2
    current_try: int = 0

    def merged_data(self, other: "ChatState") -> Dict[str, Any]:
        return merge_dicts(self.data, other.data)

    def merged_metadata(self, other: "ChatState") -> Dict[str, Any]:
        return merge_dicts(self.metadata, other.metadata)


class HCResult(BaseModel):
    binary_score: str
    explanation: Optional[str] = None


class AVResult(BaseModel):
    relevance_score: str
    explanation: Optional[str] = None


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
