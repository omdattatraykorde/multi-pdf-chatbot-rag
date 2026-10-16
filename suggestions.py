import re

from llm_factory import LLMFactory


def _clean_questions(text, limit=4):
    questions = []
    for line in text.splitlines():
        line = re.sub(r"^[\s\-\d.)]+", "", line).strip().strip('"')
        if not line:
            continue
        if "?" not in line:
            continue
        line = line[:180].strip()
        if line not in questions:
            questions.append(line)
        if len(questions) >= limit:
            break
    return questions


def _fallback_questions(vector_db, limit=4):
    docs = list(getattr(getattr(vector_db, "docstore", None), "_dict", {}).values())
    text = " ".join(doc.page_content for doc in docs[:10])
    lower = text.lower()
    visual_types = {
        doc.metadata.get("content_type")
        for doc in docs
        if doc.metadata.get("is_visual")
    }

    questions = []
    if visual_types:
        if "diagram" in visual_types:
            questions.append("What does the diagram show?")
        if "chart" in visual_types:
            questions.append("What trend does the chart or graph show?")
        if "table" in visual_types:
            questions.append("What are the key values in the table?")
        questions.append("What information is shown in the visual content?")
        if len(questions) >= limit:
            return questions[:limit]
    if any(word in lower for word in ["experience", "education", "skills", "resume", "work"]):
        questions.extend([
            "What are the main skills and areas of experience described in the document?",
            "What education or qualifications are mentioned?",
            "What are the most important projects or achievements described?",
            "Can you summarize the candidate's profile in a few points?",
        ])
    elif any(word in lower for word in ["dataset", "variable", "regression", "classification", "model"]):
        questions.extend([
            "What are the main variables, methods, or models discussed in the document?",
            "What are the most important findings or observations?",
            "What evaluation metrics or criteria are mentioned?",
            "What conclusions can be drawn from the document?",
        ])
    else:
        questions.extend([
            "What is the main objective of this document?",
            "What are the most important topics discussed?",
            "What key facts or findings should I know?",
            "Can you summarize the document in simple terms?",
        ])
    return questions[:limit]


def generate_suggested_questions(vector_db, model_name, limit=4):
    """Generate document-grounded starter questions once after processing."""
    docs = list(getattr(getattr(vector_db, "docstore", None), "_dict", {}).values())
    if not docs:
        return _fallback_questions(vector_db, limit)

    # Keep the prompt small while sampling every uploaded PDF fairly.
    by_source = {}
    for doc in docs:
        source = doc.metadata.get("source", "Unknown PDF")
        by_source.setdefault(source, []).append(doc)
    selected = []
    for source_docs in by_source.values():
        selected.extend(source_docs[:2])
    selected = selected[:10]
    visual_types = sorted({
        doc.metadata.get("content_type")
        for doc in docs
        if doc.metadata.get("is_visual") and doc.metadata.get("content_type")
    })
    context = "\n\n".join(
        f"[{doc.metadata.get('source', 'PDF')}]\n{doc.page_content[:1200]}"
        for doc in selected
    )

    prompt = f"""
Create {limit} useful starter questions for a PDF chatbot.

Rules:
- Questions must be answerable from the document excerpts below.
- Make each question specific to the uploaded material.
- Prefer different question types: overview, important details, comparison/analysis, and practical takeaway.
- Visual content types available: {", ".join(visual_types) or "none"}.
- Add a visual question only when the corresponding visual content type is available.
- For diagrams, consider asking what the diagram shows or how its flow works.
- For charts, consider asking about trends or highest/lowest points.
- For tables, consider asking about key values or comparisons.
- Do not mention 'the PDF' or 'the document' unless necessary.
- Return only a numbered list of questions, one question per line.

Document excerpts:
{context}
"""

    try:
        llm = LLMFactory.get_llm(model_name)
        response = llm.invoke(prompt)
        questions = _clean_questions(response.content, limit)
        if len(questions) >= 2:
            return questions
    except Exception as exc:
        error = str(exc).lower()
        quota_error = any(token in error for token in ["429", "quota", "resource_exhausted", "rate limit"])
        if quota_error and model_name != "Groq GPT-OSS 20B":
            try:
                llm = LLMFactory.get_llm("Groq GPT-OSS 20B")
                response = llm.invoke(prompt)
                questions = _clean_questions(response.content, limit)
                if len(questions) >= 2:
                    return questions
            except Exception:
                pass

    return _fallback_questions(vector_db, limit)
