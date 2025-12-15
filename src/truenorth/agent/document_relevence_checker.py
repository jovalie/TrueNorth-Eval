import asyncio
from typing import List, Optional
from langchain_core.prompts import PromptTemplate
from .state import ChatState
from truenorth.utils.logging import get_caller_logger
from truenorth.utils.llm import call_llm
from truenorth.utils.citation_manager import CitationManager
from pydantic import BaseModel, Field

logger = get_caller_logger()


class DocumentGrade(BaseModel):
    """Evaluation of a single document."""

    relevant: bool = Field(description="True if the document contains information relevant to the user question.")
    quote: Optional[str] = Field(description="A verbatim quote from the document that answers the question. Required if relevant.")


class BatchRelevanceGrade(BaseModel):
    """List of grades for multiple documents."""

    grades: List[DocumentGrade] = Field(description="List of grades corresponding to the order of documents provided.")


# Batch document prompt
batch_relevance_grader_prompt_template = PromptTemplate.from_template(
    """
You are a relevance grader evaluating a list of retrieved documents to see if they are helpful in answering a user question.

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
- The list order MUST match the document order (Doc 1 -> index 0).
"""
)


async def check_relevance(state: ChatState) -> ChatState:
    logger.info("---CHECK DOCUMENT RELEVANCE (BATCH MODE WITH QUOTES)---")

    if not state.documents:
        logger.info("---NO DOCUMENTS AVAILABLE, WEB SEARCH TRIGGERED---")
        state.metadata["relevance_score"] = "fail"
        return state

    question = state.question
    documents = state.documents
    model_name = state.metadata.get("model_name")
    model_provider = state.metadata.get("model_provider")

    # Format documents for the prompt
    formatted_docs = []
    for i, doc in enumerate(documents, 1):
        content = doc.page_content.replace("\n", " ").strip()[:800]  # Increased context for better quotes
        formatted_docs.append(f"[Doc {i}] {content}...")

    formatted_docs_str = "\n\n".join(formatted_docs)

    prompt_text = batch_relevance_grader_prompt_template.format(question=question, formatted_documents=formatted_docs_str)

    try:
        result = await call_llm(
            prompt=[prompt_text],
            model_name="gemini-2.0-flash",  # Use higher capacity model for extraction
            model_provider=model_provider,
            pydantic_model=BatchRelevanceGrade,
            agent_name="relevance_grader_batch",
            max_retries=2,
            verbose=True,
        )
        grades_list = result.grades
    except Exception as e:
        logger.error(f"Batch grading failed: {e}. Fallback to accepting all documents.")
        # Fallback: keep all documents if grading fails
        grades_list = [DocumentGrade(relevant=True, quote="") for _ in documents]

    # Ensure we have enough grades
    if len(grades_list) < len(documents):
        grades_list.extend([DocumentGrade(relevant=True, quote="") for _ in range(len(documents) - len(grades_list))])
    grades_list = grades_list[: len(documents)]

    # Pretty logging
    logger.info("--- Document Grading Results ---")
    filtered_documents = []

    for idx, (doc, grade_obj) in enumerate(zip(documents, grades_list), start=1):
        is_relevant = grade_obj.relevant
        mark = "✔️" if is_relevant else "❌"

        if is_relevant:
            # Store the extracted quote in metadata for CitationManager
            if grade_obj.quote:
                doc.metadata["extracted_quote"] = grade_obj.quote
                logger.info(f'{mark} Doc {idx}: RELEVANT - Quote: "{grade_obj.quote[:50]}..."')
            else:
                logger.info(f"{mark} Doc {idx}: RELEVANT - No quote extracted")

            filtered_documents.append(doc)
        else:
            logger.info(f"{mark} Doc {idx}: NOT RELEVANT")

    # Determine checker result
    total_docs = len(documents)
    kept_docs = len(filtered_documents)
    filtered_out_pct = (total_docs - kept_docs) / total_docs if total_docs > 0 else 1.0
    checker_result = "pass" if filtered_out_pct < 0.5 else "fail"

    logger.info(f"Filtered out {filtered_out_pct:.1%}: {checker_result.upper()}")

    # Update state with filtered documents
    state.documents = filtered_documents
    state.metadata["relevance_score"] = checker_result

    # NEW: Use CitationManager to process documents and assign IDs
    state = CitationManager.process_documents(state)

    logger.info(f"--- Assigned {len(state.citation_registry)} citation numbers ---")
    for src in state.citation_registry.values():
        logger.info(f"[{src.source_id}] {src.author} ({src.year}) - {src.title}")

    return state
