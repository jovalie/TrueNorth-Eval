import re
import os
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, quote

from langchain_core.documents import Document
from pydantic import BaseModel

from truenorth.agent.state import ChatState, CitationSource, CitedSource
from truenorth.utils.logging import get_caller_logger

logger = get_caller_logger()

# Get API base URL from environment, default to production
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.mytruenorth.app")
# Get Frontend URL for document viewing links
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")


class Citation(BaseModel):
    """Final output model matching lib/api/types.ts"""

    id: int
    author: str
    title: str
    year: str
    page: str
    snippet: str
    url: str
    filename: str


class CitationManager:
    """
    Central utility for handling citations across the TrueNorth RAG pipeline.
    Responsible for normalizing documents, assigning IDs, and formatting contexts.
    """

    @staticmethod
    def _clean_text(text: str) -> str:
        """Removes BOM and strips whitespace."""
        if not text:
            return ""
        return text.replace("\ufeff", "").strip()

    @staticmethod
    def _clean_snippet(text: str) -> str:
        """
        Cleans text for use as a snippet/quote.
        Removes markdown headers, bolding, and excessive whitespace.
        """
        if not text:
            return ""
        # Collapse multiple spaces/newlines into single space
        text = re.sub(r"\s+", " ", text)
        # Remove markdown headers (e.g. ## Header)
        text = re.sub(r"#+\s*", "", text)
        # Remove bold/italic markers (* or _)
        text = re.sub(r"(\*\*|__|\*|_)", "", text)
        # Remove links [text](url) -> text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        return text.strip()

    @staticmethod
    def normalize_document(doc: Document) -> CitationSource:
        """
        Converts a LangChain Document into a normalized CitationSource.
        Handles both PDF (internal) and Web sources.
        """
        metadata = doc.metadata or {}
        page_content = doc.page_content or ""

        # Clean metadata extraction helper
        def get_meta(key, default=""):
            val = metadata.get(key)
            if val is None:
                return default
            return str(val).replace("\ufeff", "").strip() or default

        # 1. Determine Type and basic fields
        url_meta = get_meta("url")
        source_meta = get_meta("source")
        file_path_meta = get_meta("file_path")

        # Heuristic: If it has a URL, it's web. Unless it's an API URL we generated previously?
        # Typically retrievers set 'file_path' for PDFs and 'url' for web.
        is_web = False
        if url_meta:
            is_web = True
        elif source_meta and source_meta.startswith("http"):
            is_web = True

        # 2. Extract Fields
        author = get_meta("author", "Unknown Author")
        # Remove BibTeX curly brackets
        if author.startswith("{") and author.endswith("}"):
            author = author[1:-1]

        title = get_meta("title", "Unknown Title")

        # Year extraction: try "year", then "creationdate"
        year = get_meta("year")
        if not year:
            cdate = get_meta("creationdate", "")
            if len(cdate) >= 4:
                year = cdate[:4]
            else:
                year = "n.d."

        page = get_meta("page") or get_meta("page_number") or get_meta("page_num") or ""

        # 3. Construct URL and Filename
        if is_web:
            final_url = url_meta or source_meta

            # If title is unknown for web, try to use domain or URL
            if title == "Unknown Title":
                title = final_url

            # If author is unknown for web, try to use domain
            if author == "Unknown Author":
                try:
                    domain = urlparse(final_url).netloc
                    author = domain.replace("www.", "")
                except:
                    author = "Web Source"

            filename = "web_source"
            doc_type = "web"
        else:
            # PDF Logic
            # Construct API URL
            file_path = file_path_meta or source_meta
            filename = os.path.basename(file_path) if file_path else "unknown.pdf"

            # Ensure page param is present
            page_val = page if page else "1"
            
            # Use frontend document viewer URL
            # Format: {FRONTEND_URL}/document?file={filename}&page={page}
            encoded_filename = quote(filename)
            final_url = f"{FRONTEND_URL}/document?file={encoded_filename}&page={page_val}"

            doc_type = "pdf"

        # 4. Content Snippet (Fallback)
        # Create a clean snippet from the first ~200 chars
        clean_content = CitationManager._clean_snippet(page_content)
        snippet = clean_content[:200]
        if len(clean_content) > 200:
            snippet += "..."

        return CitationSource(source_id=0, type=doc_type, author=author, title=title, year=year, page=page, url=final_url, filename=filename, content_snippet=snippet, full_content=page_content)  # Placeholder, assigned later

    @staticmethod
    def process_documents(state: ChatState) -> ChatState:
        """
        Ingests documents from state.documents, normalizes them, and assigns IDs.
        Updates state.citation_registry and state.documents.
        """
        documents = state.documents
        if not documents:
            return state

        registry = state.citation_registry

        # Find next available ID
        current_max_id = 0
        if registry:
            current_max_id = max(registry.keys())

        next_id = current_max_id + 1

        # Map to detect duplicates (by url + page for PDF, url for Web)
        # We need to scan existing registry to build this map
        existing_lookup = {}
        for src in registry.values():
            key = f"{src.url}"  # URL includes page for PDF
            existing_lookup[key] = src.source_id

        updated_docs = []

        for doc in documents:
            # Handle both dict and Document objects
            if isinstance(doc, dict):
                # Convert dict back to Document for normalization
                from langchain_core.documents import Document

                doc_obj = Document(page_content=doc.get("page_content", ""), metadata=doc.get("metadata", {}))
            else:
                doc_obj = doc

            # Normalize
            source = CitationManager.normalize_document(doc_obj)

            # Check duplication
            unique_key = source.url

            if unique_key in existing_lookup:
                # Reuse existing ID
                source.source_id = existing_lookup[unique_key]
            else:
                # Assign new ID
                source.source_id = next_id
                existing_lookup[unique_key] = next_id
                registry[next_id] = source
                next_id += 1

            # Update Document metadata with the assigned ID
            # Use dot access if object, dict access if dict (but we converted doc to obj above if needed)
            # However, we need to preserve the original object type in the state list if possible,
            # or just standardize to Document. Let's standardize to Document for consistency in state.

            doc_obj.metadata["source_id"] = source.source_id
            doc_obj.metadata["citation_num"] = source.source_id  # Legacy compatibility

            updated_docs.append(doc_obj)

        state.documents = updated_docs
        state.citation_registry = registry

        return state

    @staticmethod
    def get_context_string(state: ChatState) -> str:
        """
        Generates the formatted string for the LLM system prompt.
        """
        registry = state.citation_registry
        if not registry:
            return "No sources available."

        context_parts = []

        # Sort by ID
        sorted_sources = sorted(registry.values(), key=lambda x: x.source_id)

        for src in sorted_sources:
            # Format: Source [1]: Author (Year). Title.
            header = f"Source [{src.source_id}]:"
            meta_str = f"{src.author} ({src.year}). {src.title}."
            if src.page:
                meta_str += f" p.{src.page}"

            content = src.full_content.strip()

            block = f"{header}\nMetadata: {meta_str}\nContent: {content}\n"
            context_parts.append(block)

        return "\n---\n".join(context_parts)

    @staticmethod
    def resolve_citations(state: ChatState) -> Tuple[List[Dict[str, Any]], str]:
        """
        Merges LLM generated quotes with registry metadata to produce final API output.
        Renumbers citations sequentially (1, 2, 3) based on the order they are used.
        Returns:
            - List of citation dicts (matching Citation Pydantic model)
            - Renumbered response text
        """
        registry = state.citation_registry
        llm_citations = state.generated_citations
        response_text = state.generation or ""

        final_citations = []

        # 1. Gather all IDs used in structured citations or text
        used_ids = set()
        for cited in llm_citations:
            used_ids.add(cited.source_id)

        # Scan text for [N] markers as a backup
        text_ids = set(int(x) for x in re.findall(r"\[(\d+)\]", response_text))
        used_ids.update(text_ids)

        # 2. Create Map: Old ID -> New ID
        # We sort used_ids to map them sequentially: [2, 4] -> [1, 2]
        sorted_used_ids = sorted(list(used_ids))
        id_map = {old_id: new_id for new_id, old_id in enumerate(sorted_used_ids, start=1)}

        # 3. Rewrite Text: Replace [old_id] with [new_id]
        def replace_match(match):
            old_id = int(match.group(1))
            if old_id in id_map:
                return f"[{id_map[old_id]}]"
            return match.group(0)

        renumbered_text = re.sub(r"\[(\d+)\]", replace_match, response_text)

        # 4. Build Citation Objects with New IDs
        # Group quotes by source ID to pick the best one
        quotes_map = {}
        for cited in llm_citations:
            if cited.source_id not in quotes_map:
                quotes_map[cited.source_id] = cited.quote

        for old_id in sorted_used_ids:
            if old_id not in registry:
                continue

            new_id = id_map[old_id]
            source = registry[old_id]

            # Get quote
            quote = quotes_map.get(old_id, "")
            snippet = CitationManager._clean_snippet(quote)
            if not snippet or len(snippet) < 10:
                snippet = source.content_snippet

            cit = Citation(id=new_id, author=source.author, title=source.title, year=source.year, page=source.page, snippet=snippet, url=source.url, filename=source.filename)  # NEW ID
            final_citations.append(cit)

        return [c.model_dump() for c in final_citations], renumbered_text
