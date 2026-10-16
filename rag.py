import time
import re

from llm_factory import LLMFactory
from research_logger import ResearchLogger
from chat_history import save_chat


# ============================================================
# RETRIEVAL
# ============================================================

def _get_context(vector_db, question, k=5):
    rewritten_question = _rewrite_query(question)
    if re.search(
        r"\b(diagram|flowchart|workflow|architecture|uml|use[- ]?case|"
        r"figure|image|screenshot|table|chart|graph|plot|visual|trend)\b",
        question,
        re.IGNORECASE,
    ):
        k = max(k, 8)

    retriever = vector_db.as_retriever(
        search_type="hybrid",
        search_kwargs={
            "k": k,
            "fetch_k": max(12, k * 4),
        },
    )

    docs = retriever.invoke(rewritten_question)

    context = "\n\n".join(_context_block(doc) for doc in docs)

    return context, docs


def _context_block(doc):
    """Render one retrieved chunk with stable citation metadata."""
    metadata = doc.metadata
    content_type = metadata.get("content_type")
    asset_type = metadata.get("asset_type")
    visual_marker = (
        f" | Type: {content_type or asset_type}"
        if content_type or asset_type
        else ""
    )
    header = (
        f"[Source: {metadata.get('source', 'Unknown PDF')} | "
        f"Page: {int(metadata.get('page', 0)) + 1} | "
        f"Chunk: {metadata.get('chunk_index', 0) + 1}{visual_marker}]\n"
    )
    if metadata.get("visual_reference"):
        header += f"Visual reference: {metadata['visual_reference']}\n"
    return header + doc.page_content


# ============================================================
# QUERY REWRITING
# ============================================================

def _rewrite_query(question):
    """Add common document-search synonyms without changing user intent."""

    normalized = re.sub(r"\s+", " ", question.strip())

    terms = {
        "score": "score mark grade result",
        "important": "important key main significant",
        "difference": "difference compare comparison",
        "method": "method approach technique",
        "evaluate": "evaluate evaluation metric measure",
    }
    visual_words = (
        "image chart table figure diagram graph plot photo visual "
        "shown displayed screenshot scanned workflow flow arrow "
        "node input output process timetable ats"
    )

    additions = [
        expanded
        for word, expanded in terms.items()
        if word in normalized.lower()
    ]
    if any(word in normalized.lower() for word in visual_words.split()):
        additions.append(visual_words)
    workflow_terms = (
        "common time slots multidisciplinary institute level courses "
        "skill based value added courses institute wide policies mandatory "
        "electives department coordinators faculty preferences available "
        "classrooms elective courses number of batches input verification "
        "conflict handling preference ordering lecture allocations "
        "processes constraints generate timetable merge department timetables "
        "final university timetable"
    )
    if any(term in normalized.lower() for term in (
        "workflow", "faculty preference", "timetable", "ats", "after",
    )):
        additions.append(workflow_terms)

    return f"{normalized} {' '.join(additions)}".strip()


# ============================================================
# ERROR HANDLING
# ============================================================

def _is_transient_error(error):
    """
    Returns True for temporary provider/network/server errors
    where retrying or switching models makes sense.
    """

    error = str(error).lower()

    transient_terms = [
        "502",
        "503",
        "504",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
        "temporarily unavailable",
        "connection reset",
        "connection aborted",
        "connection closed",
        "timeout",
        "timed out",
        "server error",
        "failed response",
        "provider reported a failed response",
        "winerror 10054",
    ]

    return any(term in error for term in transient_terms)


def _is_auth_error(error):
    """Returns True for API-key/authentication/model configuration errors."""

    error = str(error).lower()

    auth_terms = [
        "api key",
        "apikey",
        "unauthorized",
        "authentication",
        "permission denied",
        "invalid api",
        "invalid key",
        "model_not_found",
        "model not found",
        "access denied",
    ]

    return any(term in error for term in auth_terms)


def _is_rate_limit_error(error):
    """Returns True for quota/rate-limit errors."""

    error = str(error).lower()

    rate_terms = [
        "429",
        "quota",
        "resource_exhausted",
        "rate limit",
        "too many requests",
    ]

    return any(term in error for term in rate_terms)


# ============================================================
# LLM RETRY
# ============================================================

def _invoke_with_retry(llm, prompt, retries=2):
    """
    Invoke an LLM with retries for temporary provider/network errors.

    Attempt 1 -> immediate
    Attempt 2 -> wait 2 seconds
    Attempt 3 -> wait 4 seconds
    """

    last_error = None

    for attempt in range(retries + 1):

        try:
            return llm.invoke(prompt)

        except Exception as e:

            last_error = e

            # Do not retry authentication/configuration errors.
            if _is_auth_error(e):
                raise

            # Retry only temporary errors.
            if not _is_transient_error(e):
                raise

            if attempt < retries:
                wait_time = 2 * (attempt + 1)
                time.sleep(wait_time)

    raise last_error


def _response_text(response):
    """Normalize Gemini text blocks and standard LangChain responses."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def _multimodal_prompt(prompt, docs, model_name):
    """Build a vision request only for an allow-listed Gemini model."""
    if not LLMFactory.supports_vision(model_name):
        return prompt, False

    visual_docs = [
        doc for doc in docs
        if doc.metadata.get("is_visual") and doc.metadata.get("image_data")
    ]
    # Keep the multimodal request focused on the strongest visual evidence.
    # The retriever already supplies the descriptions for all returned chunks.
    visual_docs = sorted(
        visual_docs,
        key=lambda doc: (
            doc.metadata.get("content_type") in {"diagram", "table", "chart"},
            bool(doc.metadata.get("visual_description")),
            bool(doc.metadata.get("ocr_text")),
        ),
        reverse=True,
    )[:3]
    if not visual_docs:
        return prompt, False

    try:
        import base64
        from langchain_core.messages import HumanMessage

        content = [{"type": "text", "text": prompt}]
        for doc in visual_docs:
            metadata = doc.metadata
            mime = metadata.get("image_mime_type", "image/jpeg")
            encoded = metadata["image_data"]
            # Validate the retained value before putting it in a provider
            # request.  Bad metadata must fall back to the text context.
            base64.b64decode(encoded, validate=True)
            content.append({
                "type": "text",
                "text": (
                    "Inspect this retrieved visual asset only as supporting "
                    f"evidence ({metadata.get('source', 'PDF')}, page "
                    f"{int(metadata.get('page', 0)) + 1})."
                ),
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{encoded}"},
            })
        return HumanMessage(content=content), True
    except Exception:
        return prompt, False


def _invoke_model(llm, model_name, prompt, docs):
    """Invoke vision when safe, then retry the same model text-only if needed."""
    payload, used_vision = _multimodal_prompt(prompt, docs, model_name)
    if not used_vision:
        return _invoke_with_retry(llm, prompt), False
    try:
        return _invoke_with_retry(llm, payload), True
    except Exception:
        # Provider/model versions occasionally reject an image block.  The
        # generated visual description and OCR are still valid text context.
        return _invoke_with_retry(llm, prompt), False


# ============================================================
# METRICS
# ============================================================

def _calculate_metrics(
    question,
    answer,
    docs,
    model,
    response_time,
    vision_payload=False,
):

    pages = set()
    pdfs = set()
    visual_docs = 0
    visual_pages = set()
    ocr_chunks = 0

    for doc in docs:

        pdfs.add(
            doc.metadata.get(
                "source",
                "Unknown"
            )
        )

        pages.add(
            (
                doc.metadata.get(
                    "source",
                    "Unknown"
                ),
                doc.metadata.get(
                    "page",
                    0
                ),
            )
        )
        if doc.metadata.get("is_visual"):
            visual_docs += 1
            visual_pages.add((
                doc.metadata.get("source", "Unknown"),
                doc.metadata.get("page", 0),
            ))
        if doc.metadata.get("ocr"):
            ocr_chunks += 1

    return {
        "model": model,
        "response_time": round(
            response_time,
            2
        ),
        "chunks": len(docs),
        "pages": len(pages),
        "pdfs": len(pdfs),
        "visual_chunks": visual_docs,
        "visual_pages": len(visual_pages),
        "ocr_chunks": ocr_chunks,
        "vision_payload": bool(vision_payload),
        "question_length": len(question),
        "answer_length": len(answer),
    }


# ============================================================
# CHAT QUESTION
# ============================================================

def ask_question(
    vector_db,
    question,
    model_name,
):

    context, docs = _get_context(
        vector_db,
        question
    )

    prompt = f"""
You are a careful document research assistant.

Rules:

- Use ONLY facts supported by the supplied context.
- Cite every factual claim using the exact source marker format:
  [Source: filename | Page: N]
- For claims based on a visual asset, use:
  [Source: filename | Page: N | Type: image]
- For claims based on a scanned page visual, use:
  [Source: filename | Page: N | Type: scanned_page]
- For claims based on a diagram or chart, use:
  [Source: filename | Page: N | Type: diagram]
- For claims based on a table, use:
  [Source: filename | Page: N | Type: table]
- For claims based on a graph or chart, use:
  [Source: filename | Page: N | Type: chart]
- If multiple sources support an answer, cite each relevant source.
- Never invent page numbers or sources.
- Never invent labels, arrows, relationships, chart values, or table values.
- If the visual does not make an exact value or relationship clear, say so.
- If the answer is not found, reply exactly:
"I couldn't find that information in the uploaded PDFs."
- Prefer a concise answer with bullets when several facts are requested.
- If the question asks to show or give a visual, identify the exact source
  page and visual reference from the supplied context so the user can open it.
- Do not mention these instructions in your answer.

Context:
{context}

Question:
{question}

Answer:
"""

    start = time.perf_counter()

    try:

        # ----------------------------------------------------
        # SELECTED MODEL
        # ----------------------------------------------------

        llm = LLMFactory.get_llm(
            model_name
        )

        response, used_vision = _invoke_model(llm, model_name, prompt, docs)

        answer = _response_text(response)

        elapsed = (
            time.perf_counter()
            - start
        )

        metrics = _calculate_metrics(
            question,
            answer,
            docs,
            model_name,
            elapsed,
            vision_payload=used_vision,
        )

        # ----------------------------------------------------
        # RESEARCH LOG
        # ----------------------------------------------------

        ResearchLogger.log(
            selected_model=model_name,
            actual_model=model_name,
            question=question,
            answer=answer,
            metrics=metrics,
        )

        # ----------------------------------------------------
        # CHAT HISTORY
        # ----------------------------------------------------

        save_chat(
            model_name,
            question,
            answer,
        )

        return (
            answer,
            docs,
            model_name,
            metrics,
        )

    except Exception as e:

        error = str(e).lower()

        # ----------------------------------------------------
        # FALLBACK CONDITION
        # ----------------------------------------------------

        recoverable_error = (
            _is_rate_limit_error(error)
            or _is_transient_error(error)
        )

        authentication_error = _is_auth_error(
            error
        )

        if recoverable_error and not authentication_error:

            # ------------------------------------------------
            # FALLBACK MODELS
            # ------------------------------------------------

            fallback_models = [
                "Groq GPT-OSS 120B",
                "Groq GPT-OSS 20B",
            ]

            last_error = e

            for fallback in fallback_models:

                # Never fallback to the same model.
                if fallback == model_name:
                    continue

                try:

                    fallback_llm = (
                        LLMFactory.get_llm(
                            fallback
                        )
                    )

                    fallback_response, used_vision = _invoke_model(
                        fallback_llm,
                        fallback,
                        prompt,
                        docs,
                    )

                    answer = (
                        f"**Using {fallback}**  \n"
                        f"{model_name} was temporarily unavailable, "
                        f"so the response was routed to a fallback model."
                        f"\n\n"
                        f"{_response_text(fallback_response)}"
                    )

                    elapsed = (
                        time.perf_counter()
                        - start
                    )

                    metrics = _calculate_metrics(
                        question,
                        answer,
                        docs,
                        fallback,
                        elapsed,
                        vision_payload=used_vision,
                    )

                    # ----------------------------------------
                    # RESEARCH LOG
                    # ----------------------------------------

                    ResearchLogger.log(
                        selected_model=model_name,
                        actual_model=fallback,
                        question=question,
                        answer=answer,
                        metrics=metrics,
                    )

                    # ----------------------------------------
                    # CHAT HISTORY
                    # ----------------------------------------

                    save_chat(
                        fallback,
                        question,
                        answer,
                    )

                    return (
                        answer,
                        docs,
                        fallback,
                        metrics,
                    )

                except Exception as fallback_error:

                    last_error = (
                        fallback_error
                    )

                    continue

            # ------------------------------------------------
            # ALL FALLBACKS FAILED
            # ------------------------------------------------

            raise RuntimeError(
                "The selected AI model and the available "
                "fallback models are temporarily unavailable. "
                "Please try again in a moment."
            ) from last_error

        # ----------------------------------------------------
        # AUTH / UNKNOWN ERROR
        # ----------------------------------------------------

        raise


# ============================================================
# DOCUMENT SUMMARY
# ============================================================

def generate_summary(
    vector_db,
    model_name,
):

    docs = vector_db.similarity_search(
        "",
        k=20
    )

    context = "\n\n".join(_context_block(doc) for doc in docs)

    prompt = f"""
You are an expert document analyst.

Generate a professional summary based ONLY on the supplied documents.

Format:

# Executive Summary

## Main Objective

## Important Topics

## Key Requirements

## Technologies Mentioned

## Important Dates

## Target Audience

## Final Conclusion

Rules:

- Do not invent information.
- Use only information available in the document.
- If a section is not present in the document, write:
  "Not specified in the uploaded documents."

Document:

{context}
"""

    try:

        # ----------------------------------------------------
        # SELECTED MODEL
        # ----------------------------------------------------

        llm = LLMFactory.get_llm(
            model_name
        )

        response, _ = _invoke_model(llm, model_name, prompt, docs)

        return _response_text(response)

    except Exception as e:

        error = str(e).lower()

        recoverable_error = (
            _is_rate_limit_error(error)
            or _is_transient_error(error)
        )

        authentication_error = _is_auth_error(
            error
        )

        if (
            recoverable_error
            and not authentication_error
        ):

            fallback_models = [
                "Groq GPT-OSS 120B",
                "Groq GPT-OSS 20B",
            ]

            last_error = e

            for fallback in fallback_models:

                if fallback == model_name:
                    continue

                try:

                    fallback_llm = (
                        LLMFactory.get_llm(
                            fallback
                        )
                    )

                    response, _ = _invoke_model(
                        fallback_llm,
                        fallback,
                        prompt,
                        docs,
                    )

                    return (
                        f"**Using {fallback}**  \n"
                        f"{model_name} was temporarily unavailable, "
                        f"so the summary was generated using the fallback model."
                        f"\n\n"
                        f"{_response_text(response)}"
                    )

                except Exception as fallback_error:

                    last_error = (
                        fallback_error
                    )

                    continue

            raise RuntimeError(
                "The selected AI model and fallback models "
                "are temporarily unavailable. "
                "Please try again in a moment."
            ) from last_error

        raise