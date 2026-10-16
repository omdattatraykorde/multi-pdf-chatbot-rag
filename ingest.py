"""PDF ingestion for text and visual RAG.

Text remains the primary representation of a PDF.  In addition, embedded
images and rendered scanned pages become small, searchable ``Document``
objects in the same FAISS index.  Image bytes are retained in metadata so a
vision-capable Gemini model can inspect the exact retrieved asset at answer
time; non-vision models only receive the generated text description/OCR.
"""

import base64
import hashlib
import io
import os
import re
import tempfile
import time

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from llm_factory import LLMFactory
from vector_store import HybridVectorStore, get_embeddings


MAX_ASSET_BYTES = 1_500_000
MAX_IMAGES_PER_PAGE = 12
MIN_RENDER_SCALE = 1.8
MAX_RENDER_SCALE = 2.6
_PADDLE_OCR = None
_PADDLE_OCR_INITIALIZED = False
_STRUCTURE_ENGINE = None
_STRUCTURE_INITIALIZED = False
_VISUAL_CACHE = {}


def _clean_page_text(text):
    """Remove extraction noise while preserving paragraph boundaries."""
    text = (text or "").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _uploaded_bytes(uploaded_file):
    """Read Streamlit uploads and file-like test doubles without surprises."""
    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
    else:
        data = uploaded_file.read()
    if isinstance(data, str):
        data = data.encode("utf-8")
    return data or b""


def _image_bytes_for_embedding(raw_bytes, mime_type):
    """Bound retained image payloads while keeping common PDF formats intact."""
    if len(raw_bytes) <= MAX_ASSET_BYTES:
        return raw_bytes, mime_type

    try:
        from PIL import Image

        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        image.thumbnail((1800, 1800))
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=78, optimize=True)
        return output.getvalue(), "image/jpeg"
    except Exception:
        # A text representation is still useful if an unusual image cannot be
        # transcoded.  Avoid retaining an unexpectedly huge object in FAISS.
        return raw_bytes[:MAX_ASSET_BYTES], mime_type


def _render_page(page):
    """Render a readable page image without creating oversized API payloads."""
    import fitz

    max_dimension = max(float(page.rect.width), float(page.rect.height))
    scale = min(MAX_RENDER_SCALE, max(MIN_RENDER_SCALE, 2200 / max_dimension))
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        alpha=False,
    )
    raw = pixmap.tobytes("jpg", jpg_quality=82)
    return _image_bytes_for_embedding(raw, "image/jpeg")


def _ocr_image(image_bytes):
    """Extract visible text with PaddleOCR without blocking ingestion."""
    digest = hashlib.sha256(image_bytes).hexdigest()
    cached = _VISUAL_CACHE.get(("ocr", digest))
    if cached is not None:
        return cached
    global _PADDLE_OCR, _PADDLE_OCR_INITIALIZED
    if not _PADDLE_OCR_INITIALIZED:
        _PADDLE_OCR_INITIALIZED = True
        try:
            from paddleocr import PaddleOCR
            try:
                _PADDLE_OCR = PaddleOCR(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    lang="en",
                )
            except TypeError:
                _PADDLE_OCR = PaddleOCR(lang="en", use_angle_cls=True)
        except Exception:
            _PADDLE_OCR = None
    if _PADDLE_OCR is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(image_bytes)
                image_path = tmp.name
            try:
                if hasattr(_PADDLE_OCR, "predict"):
                    result = _PADDLE_OCR.predict(image_path)
                else:
                    result = _PADDLE_OCR.ocr(image_path, cls=True)
                texts = []
                for page_result in result or []:
                    data = getattr(page_result, "json", None)
                    data = data() if callable(data) else data
                    if isinstance(data, dict):
                        data = data.get("res", data)
                        texts.extend(data.get("rec_texts", []))
                    elif isinstance(page_result, list):
                        texts.extend(
                            item[1][0] for item in page_result
                            if isinstance(item, list) and len(item) > 1
                        )
                text = _clean_page_text(" ".join(str(item) for item in texts))
                if text:
                    _VISUAL_CACHE[("ocr", digest)] = text
                    return text
            finally:
                if os.path.exists(image_path):
                    os.remove(image_path)
        except Exception:
            pass
    _VISUAL_CACHE[("ocr", digest)] = ""
    return ""


def _structure_text(image_bytes):
    """Extract layout/table text when PP-StructureV3 is available."""
    digest = hashlib.sha256(image_bytes).hexdigest()
    cached = _VISUAL_CACHE.get(("structure", digest))
    if cached is not None:
        return cached
    global _STRUCTURE_ENGINE, _STRUCTURE_INITIALIZED
    if not _STRUCTURE_INITIALIZED:
        _STRUCTURE_INITIALIZED = True
        try:
            from paddleocr import PPStructureV3
            _STRUCTURE_ENGINE = PPStructureV3(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except Exception:
            _STRUCTURE_ENGINE = None
    if _STRUCTURE_ENGINE is None:
        return ""
    image_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_bytes)
            image_path = tmp.name
        result = _STRUCTURE_ENGINE.predict(image_path)
        values = []

        def collect(value):
            if isinstance(value, str):
                if value.strip():
                    values.append(value.strip())
            elif isinstance(value, dict):
                for key, item in value.items():
                    if key in {"input_path", "page_index", "img"}:
                        continue
                    collect(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)
            else:
                data = getattr(value, "json", None)
                if callable(data):
                    collect(data())

        collect(result)
        text = _clean_page_text(" ".join(values))
        _VISUAL_CACHE[("structure", digest)] = text
        return text
    except Exception:
        _VISUAL_CACHE[("structure", digest)] = ""
        return ""
    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)


def _visual_content_type(description, ocr_text, default="image"):
    """Classify visual evidence without changing the underlying asset."""
    text = f"{description} {ocr_text}".lower()
    if any(term in text for term in (
        "flowchart", "workflow", "architecture diagram", "use case",
        "uml", "arrow", "nodes", "process flow",
    )):
        return "diagram"
    if any(term in text for term in ("table", "rows", "columns", "tabular")):
        return "table"
    if any(term in text for term in (
        "chart", "graph", "x-axis", "y-axis", "trend", "legend",
        "bar", "line plot", "data point",
    )):
        return "chart"
    return default


def _meaningful_image(raw_bytes, extracted):
    """Ignore tiny decorative images while retaining likely figures/logos."""
    try:
        width = int(extracted.get("width", 0))
        height = int(extracted.get("height", 0))
        if width and height and width * height < 10_000:
            return False
    except (TypeError, ValueError):
        pass
    return bool(raw_bytes)


def _needs_structure(extracted, ocr_text):
    """Run layout analysis for likely document visuals, not tiny decorations."""
    try:
        area = int(extracted.get("width", 0)) * int(extracted.get("height", 0))
    except (TypeError, ValueError):
        area = 0
    return area >= 200_000 or len(ocr_text or "") >= 40


def _response_text(response):
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return _clean_page_text(str(content))


def _describe_image(vision_llm, image_bytes, mime_type, page_number, kind):
    """Ask Gemini for a grounded visual description, returning empty on error."""
    if vision_llm is None:
        return ""
    digest = hashlib.sha256(image_bytes).hexdigest()
    cached = _VISUAL_CACHE.get(("description", digest, kind))
    if cached is not None:
        return cached
    try:
        from langchain_core.messages import HumanMessage

        encoded = base64.b64encode(image_bytes).decode("ascii")
        prompt = (
            "Analyze this PDF visual for a searchable multimodal document index. "
            "Return a detailed, factual structured description. For a diagram "
            "or workflow, list every readable node/label, inputs, constraints, "
            "processing stages, outputs, arrows or connections, and the order "
            "in which information flows. Explain relationships such as "
            "what feeds into each stage and what each stage produces. Include "
            "important visible text exactly where possible. For tables/charts, "
            "describe headings, values, trends, and relationships. Do not "
            "invent unreadable content; mark uncertainty explicitly. Use plain "
            "text headings: Title, Visible labels, Inputs, Flow and connections, "
            "Processing, Outputs, and OCR/search terms. "
            f"This is a {kind} on page {page_number}."
        )
        response = vision_llm.invoke(HumanMessage(content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
            },
        ]))
        description = _response_text(response)
        _VISUAL_CACHE[("description", digest, kind)] = description
        return description
    except Exception:
        _VISUAL_CACHE[("description", digest, kind)] = ""
        return ""


def _vision_llm(model_name):
    """Create an optional vision client. Configuration/network failures are safe."""
    # Visual descriptions are a Gemini-only ingestion capability.  If the
    # selected chat model is Groq, use Gemini for indexing when its key is
    # configured; answering still honors the user's selected model.
    candidate = model_name if LLMFactory.supports_vision(model_name) else "Gemini 2.5 Flash"
    try:
        return LLMFactory.get_llm(candidate), candidate
    except Exception:
        return None, "text-only"


def _visual_document(
    source,
    page_index,
    image_bytes,
    mime_type,
    asset_type,
    ocr_text,
    description,
    content_type=None,
    image_index=None,
    structure_text="",
    visual_id=None,
    surrounding_text="",
):
    page_number = page_index + 1
    description = description or (
        f"{asset_type.replace('_', ' ').capitalize()} on page {page_number}. "
        "No automatic visual description is available."
    )
    parts = [
        f"Visual asset from {source}, page {page_number}.",
        f"Asset type: {asset_type.replace('_', ' ')}.",
        "Visual search terms: image chart table figure diagram graph plot.",
        f"Visual description: {description}",
    ]
    if ocr_text:
        parts.append(f"Visible text (OCR): {ocr_text}")
    if structure_text:
        parts.append(f"Structured layout text: {structure_text}")
    if surrounding_text:
        parts.append(f"Nearby page context: {surrounding_text}")
    digest = hashlib.sha256(image_bytes).hexdigest()
    return Document(
        page_content="\n".join(parts),
        metadata={
            "source": source,
            "page": page_index,
            "page_label": page_number,
            "content_type": content_type or (
                "scanned_page" if asset_type == "scanned_page" else
                _visual_content_type(description, ocr_text)
            ),
            "asset_type": asset_type,
            "is_visual": True,
            "image_index": image_index,
            "visual_id": visual_id or f"page{page_number}_visual{(image_index or 0) + 1}",
            "image_hash": digest,
            "image_mime_type": mime_type,
            "image_data": base64.b64encode(image_bytes).decode("ascii"),
            "visual_reference": f"{source}#page={page_number}",
            "visual_description": description,
            "ocr_text": ocr_text,
            "structure_text": structure_text,
            "surrounding_text": surrounding_text,
            "ocr": bool(ocr_text),
        },
    )


def _load_pdf_with_ocr(temp_path, source="Unknown PDF", model_name=None):
    """Load text, recover scanned pages, and extract visual assets robustly."""
    try:
        documents = PyMuPDFLoader(temp_path).load()
    except Exception:
        documents = []

    ocr_count = 0
    rendered_pages = 0
    image_count = 0
    visual_documents = []
    visual_descriptions = 0
    visual_description_failures = 0
    ocr_failures = 0
    visual_hashes = set()
    vision_llm, vision_backend = _vision_llm(model_name)

    try:
        import fitz
    except ImportError:
        # PyMuPDF is a required dependency, but leave a useful text-only
        # fallback for partially installed environments.
        return documents, {
            "ocr_pages": 0,
            "rendered_pages": 0,
            "images_extracted": 0,
            "visual_documents": 0,
            "visual_descriptions": 0,
            "visual_description_failures": 0,
            "ocr_failures": 0,
            "vision_backend": "text-only",
        }

    pdf = fitz.open(temp_path)
    try:
        # PyMuPDFLoader and fitz use the same zero-based page ordering.  Build
        # missing page documents when a loader version cannot parse a page.
        while len(documents) < len(pdf):
            documents.append(Document(page_content="", metadata={"page": len(documents)}))

        for page_index in range(len(pdf)):
            page = pdf.load_page(page_index)
            document = documents[page_index]
            document.metadata["source"] = source
            document.metadata["page"] = page_index
            document.metadata["page_label"] = page_index + 1

            page_text = _clean_page_text(document.page_content)
            if not page_text:
                try:
                    rendered, render_mime = _render_page(page)
                    rendered_pages += 1
                    ocr_text = _ocr_image(rendered)
                    if ocr_text:
                        document.page_content = ocr_text
                        document.metadata["ocr"] = True
                        ocr_count += 1
                    else:
                        ocr_failures += 1
                    # A scanned page is also retained as a visual asset so
                    # charts/forms remain available even when OCR is weak.
                    description = _describe_image(
                        vision_llm, rendered, render_mime, page_index + 1, "scanned page"
                    )
                    if description:
                        visual_descriptions += 1
                    elif vision_llm is not None:
                        visual_description_failures += 1
                    visual_documents.append(_visual_document(
                        source, page_index, rendered, render_mime,
                        "scanned_page", ocr_text, description,
                        content_type="scanned_page",
                    ))
                except Exception:
                    # Keep the page as an empty text document; any other pages
                    # and extracted images must still be indexed.
                    pass

            # Vector diagrams often have selectable labels but no raster image.
            # Render pages with substantial drawing content so the complete
            # visual structure (including arrows and relationships) is indexed.
            page_lower = page_text.lower()
            try:
                drawing_count = len(page.get_drawings())
            except Exception:
                drawing_count = 0
            try:
                page_image_count = len(page.get_images(full=True))
            except Exception:
                page_image_count = 0
            diagram_signal = (
                drawing_count >= 8
                or page_image_count > 0 and len(page_text) >= 120
                or any(term in page_lower for term in (
                    "diagram", "flowchart", "workflow", "figure", "chart",
                    "graph", "table", "screenshot", "architecture", "use case",
                ))
            )
            if page_text and diagram_signal:
                try:
                    rendered, render_mime = _render_page(page)
                    rendered_pages += 1
                    rendered_ocr = _ocr_image(rendered)
                    if rendered_ocr:
                        ocr_count += 1
                    description = _describe_image(
                        vision_llm, rendered, render_mime, page_index + 1,
                        "full-page vector diagram",
                    )
                    if description:
                        visual_descriptions += 1
                    elif vision_llm is not None:
                        visual_description_failures += 1
                    structure_text = _structure_text(rendered)
                    full_page_type = _visual_content_type(
                        description, f"{rendered_ocr} {page_text}", "figure"
                    )
                    visual_doc = _visual_document(
                        source, page_index, rendered, render_mime, "page_render",
                        rendered_ocr, description, content_type=full_page_type,
                        structure_text=structure_text,
                        visual_id=f"page{page_index + 1}_full_page",
                        surrounding_text=page_text[:1800],
                    )
                    visual_doc.metadata["content_type"] = "diagram"
                    visual_doc.page_content = (
                        f"{visual_doc.page_content}\n"
                        f"Extracted page text: {page_text}"
                    )
                    visual_documents.append(visual_doc)
                except Exception:
                    pass

            seen_hashes = set()
            try:
                page_images = page.get_images(full=True)
            except Exception:
                page_images = []
            # ``get_images`` does not expose every inline image on some PDF
            # producers.  Add xrefs reported by the richer image-info API.
            try:
                known_xrefs = {item[0] for item in page_images if item}
                for image_info in page.get_image_info(xrefs=True):
                    xref = image_info.get("xref")
                    if xref and xref not in known_xrefs:
                        page_images.append((xref,))
                        known_xrefs.add(xref)
            except Exception:
                pass
            for image_position, image_info in enumerate(page_images[:MAX_IMAGES_PER_PAGE]):
                try:
                    xref = image_info[0]
                    extracted = pdf.extract_image(xref)
                    raw = extracted.get("image", b"")
                    if not _meaningful_image(raw, extracted):
                        continue
                    digest = hashlib.sha256(raw).hexdigest()
                    if digest in visual_hashes:
                        continue
                    seen_hashes.add(digest)
                    visual_hashes.add(digest)
                    image_count += 1
                    mime = extracted.get("ext", "png").lower()
                    mime = f"image/{'jpeg' if mime in ('jpg', 'jpeg') else mime}"
                    image_bytes, mime = _image_bytes_for_embedding(raw, mime)
                    ocr_text = _ocr_image(image_bytes)
                    structure_text = (
                        _structure_text(image_bytes)
                        if _needs_structure(extracted, ocr_text)
                        else ""
                    )
                    description = _describe_image(
                        vision_llm, image_bytes, mime, page_index + 1, "embedded image"
                    )
                    if description:
                        visual_descriptions += 1
                    elif vision_llm is not None:
                        visual_description_failures += 1
                    visual_documents.append(_visual_document(
                        source, page_index, image_bytes, mime,
                        "embedded_image", ocr_text, description,
                        image_index=image_position,
                        structure_text=structure_text,
                        visual_id=f"page{page_index + 1}_image{image_position + 1}",
                        surrounding_text=page_text[:1200],
                    ))
                except Exception:
                    continue
    finally:
        pdf.close()

    documents.extend(visual_documents)
    return documents, {
        "ocr_pages": ocr_count,
        "rendered_pages": rendered_pages,
        "images_extracted": image_count,
        "visual_documents": len(visual_documents),
        "visual_descriptions": visual_descriptions,
        "visual_description_failures": visual_description_failures,
        "ocr_failures": ocr_failures,
        "vision_backend": vision_backend,
    }


def _chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=140,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
        add_start_index=True,
    )
    chunks = []
    for page_document in documents:
        page_document.page_content = _clean_page_text(page_document.page_content)
        if not page_document.page_content:
            continue
        page_chunks = splitter.split_documents([page_document])
        for chunk_index, chunk in enumerate(page_chunks):
            chunk.metadata["chunk_index"] = chunk_index
            chunk.metadata["page_label"] = int(chunk.metadata.get("page", 0)) + 1
            chunks.append(chunk)
    return chunks


def create_vector_db(uploaded_files, model_name=None):
    """Index text, OCR, and visual descriptions in one hybrid FAISS store."""
    start_time = time.perf_counter()
    documents = []
    totals = {
        "ocr_pages": 0,
        "rendered_pages": 0,
        "images_extracted": 0,
        "visual_documents": 0,
        "visual_descriptions": 0,
        "visual_description_failures": 0,
        "ocr_failures": 0,
    }
    total_pages = 0
    vision_backend = "text-only"

    for uploaded_file in uploaded_files:
        source = getattr(uploaded_file, "name", "uploaded.pdf")
        data = _uploaded_bytes(uploaded_file)
        temp_path = None
        docs = []
        stats = {key: 0 for key in totals}
        stats["vision_backend"] = "text-only"
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(data)
                temp_path = tmp.name
            try:
                docs, stats = _load_pdf_with_ocr(temp_path, source, model_name)
            except Exception:
                # A corrupt PDF should not discard other uploaded documents.
                docs = []
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        for key in totals:
            totals[key] += stats.get(key, 0)
        vision_backend = stats.get("vision_backend", vision_backend)
        total_pages += max(
            [int(doc.metadata.get("page", 0)) for doc in docs] + [-1]
        ) + 1
        documents.extend(docs)

    chunks = _chunk_documents(documents)
    if not chunks:
        raise ValueError(
            "No readable text or visual content was found in the uploaded PDFs. "
            "For scanned files, install Tesseract OCR or configure a Gemini API key."
        )

    embeddings, embedding_backend = get_embeddings()
    faiss_store = FAISS.from_documents(chunks, embeddings)
    vector_db = HybridVectorStore(faiss_store, chunks)
    embedding_time = time.perf_counter() - start_time

    experiment_data = {
        "total_pdfs": len(uploaded_files),
        "total_pages": total_pages,
        "total_chunks": len(chunks),
        "ocr_pages": totals["ocr_pages"],
        "rendered_pages": totals["rendered_pages"],
        "images_extracted": totals["images_extracted"],
        "visual_documents": totals["visual_documents"],
        "visual_descriptions": totals["visual_descriptions"],
        "visual_description_failures": totals["visual_description_failures"],
        "ocr_failures": totals["ocr_failures"],
        "vision_backend": vision_backend,
        "retrieval": "hybrid dense + lexical rank fusion (text + visual)",
        "chunking": "page-aware semantic separators with overlap",
        "embedding_backend": embedding_backend,
        "embedding_time": round(embedding_time, 3),
    }
    return vector_db, experiment_data
