import asyncio
from typing import List
from langchain_core.prompts import PromptTemplate
from .state import ChatState
from truenorth.utils.logging import get_caller_logger
from truenorth.utils.llm import call_llm
from truenorth.utils.citation_manager import CitationManager
from pydantic import BaseModel, Field

logger = get_caller_logger()


class BatchRelevanceGrade(BaseModel):
    """List of boolean grades for multiple documents."""

    grades: List[bool] = Field(description="List of true/false grades corresponding to the order of documents provided.")


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
- Return a JSON object with a single key `grades` which is a list of booleans.
- Use `true` if the document contains related or helpful information.
- Use `false` if completely unrelated.
- The list order MUST match the document order (Doc 1 -> index 0).
"""
)


async def check_relevance(state: ChatState) -> ChatState:
    logger.info("---CHECK DOCUMENT RELEVANCE (BATCH MODE)---")

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
        content = doc.page_content.replace("\n", " ").strip()[:500]  # Truncate for efficiency
        formatted_docs.append(f"[Doc {i}] {content}...")

    formatted_docs_str = "\n\n".join(formatted_docs)

    prompt_text = batch_relevance_grader_prompt_template.format(question=question, formatted_documents=formatted_docs_str)

    try:
        result = await call_llm(
            prompt=[prompt_text],
            model_name='gemini-2.5-flash'',
            model_provider=model_provider,
            pydantic_model=BatchRelevanceGrade,
            agent_name="relevance_grader_batch",
            max_retries=2,
            verbose=True,
        )
        grades = result.grades
    except Exception as e:
        logger.error(f"Batch grading failed: {e}. Fallback to accepting all documents.")
        # Fallback: keep all documents if grading fails
        grades = [True] * len(documents)

    # Ensure we have enough grades (pad with True if short, slice if long)
    if len(grades) < len(documents):
        grades.extend([True] * (len(documents) - len(grades)))
    grades = grades[: len(documents)]

    # Pretty logging
    logger.info("--- Document Grading Results ---")
    filtered_documents = []
    for idx, (doc, grade) in enumerate(zip(documents, grades), start=1):
        mark = "✔️" if grade else "❌"
        snippet = doc.page_content.strip().replace("\n", " ")[:100]
        logger.info(f'{mark} Document {idx}: {grade} - "{snippet}..."')
        if grade:
            filtered_documents.append(doc)

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
