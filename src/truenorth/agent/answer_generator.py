import time
import os
from datetime import datetime
from typing import List, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from truenorth.utils.llm import call_llm
from truenorth.agent.state import show_agent_reasoning, CitedSource
from truenorth.utils.logging import get_caller_logger
from truenorth.utils.metaprompt import goals_as_str
from truenorth.utils.citation_manager import CitationManager

logger = get_caller_logger()


class AnswerResponse(BaseModel):
    answer: str = Field(description="The natural language answer to the user's question, including [ID] citations.")
    # We no longer need the LLM to extract citations/quotes, we infer them from [ID] tags
    citations: Optional[List[CitedSource]] = Field(default_factory=list, description="Deprecated. Leave empty.")


# Define the structured prompt template for answer generation
answer_generator_prompt_template = PromptTemplate.from_template(
    """
Today is {current_datetime}.
                                                                
You are an assistant for question-answering tasks.

Here are your goals:
{goals_as_str}

You do not replace a therapist, legal counsel, or HR department, but you can provide emotional support, educational context, helpful language, and confidential documentation tools.

**Available Sources for Citation:**
{source_context}

Use the above numbered sources to support your answer. When referencing information, use the format [ID] (e.g., [1], [2]) in the text.

---

**Background Knowledge**:
The available sources above provide the necessary background knowledge.

**User Question**:
{question}
                                                                
---
                                                                
**Instructions**:
1. Base your answer primarily on the background knowledge provided above.
2. Use numbered citations in the text when referencing specific information (e.g., [1], [2]).
3. If the answer is **not present** in the knowledge, say so explicitly.
4. Be **comprehensive**, **accurate**, and **focused**. Do not give short, generic answers.
5. Provide **concrete, actionable advice** that an individual can use to improve their situation, not an organizational solution.
6. Only answer questions relevant to STEM, workplace support, or academic guidance.
7. Return your response in the specified structured format (answer text only).

---
**Important**:
- Never invent or guess answers using general world knowledge.  
- Your role is to **maintain trust** and offer emotionally supportive, mission-aligned responses.
- Always keep a lighthearted but **thorough** manner of speaking while providing a helpful answer to the question.
"""
)


# ------------------------ Answer Generator Node ------------------------
def answer_generator(state):
    """
    Generates an answer based on the retrieved documents and user question.
    """
    logger.info("\n---ANSWER GENERATION---")

    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Use original_question if available
    if state.original_question:
        question = state.original_question
    else:
        question = state.question

    # Create citation context using Manager
    source_context = CitationManager.get_context_string(state)

    # Format the prompt
    prompt = answer_generator_prompt_template.format(current_datetime=current_datetime, question=question, goals_as_str=goals_as_str, source_context=source_context)

    logger.info(f"Answer generator prompt length: {len(prompt)}")

    # Force use of Gemini 2.5 Pro for answer generation
    # While preserving other state metadata
    answer_model_name = os.getenv("MODEL_NAME_THINK", "gpt-5")

    # Call LLM with structured output using Gemini 2.5
    response_obj = call_llm(prompt=prompt, model_name=answer_model_name, model_provider=os.getenv("MODEL_PROVIDER", "OpenAI"), pydantic_model=AnswerResponse, agent_name="answer_generator_agent")

    show_agent_reasoning(response_obj, f"Answer Generator Response | " + answer_model_name)

    # Extract components
    if isinstance(response_obj, AnswerResponse):
        generation_text = response_obj.answer
    else:
        # Fallback if structured output failed
        generation_text = str(response_obj.content) if hasattr(response_obj, "content") else str(response_obj)

    # Store the clean answer
    state.generation = generation_text

    # Populate generated_citations based on used IDs in the text
    # The actual resolution happens in CitationManager.resolve_citations,
    # but we can populate state.generated_citations for completeness if needed elsewhere.
    # However, CitationManager logic now relies on the registry quotes, so we don't strictly need structured_citations from LLM.
    state.generated_citations = []

    # Add message to history
    state.messages.append(AIMessage(content=generation_text))

    logger.info(f"Response: {state.generation[:200]}...")

    return state
