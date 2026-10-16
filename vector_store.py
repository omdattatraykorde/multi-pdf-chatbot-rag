"""Hybrid retrieval for the local document index.

The FAISS index remains the durable vector store, while a small in-memory
lexical index handles exact names, numbers, and acronyms that embeddings
often miss. Results are fused by reciprocal rank, which is deterministic and
does not require a second hosted service.
"""

import re
import hashlib
from collections import Counter, defaultdict

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import numpy as np


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*", re.IGNORECASE)
_VISUAL_QUERY_RE = re.compile(
    r"\b(diagram|flowchart|workflow|architecture|uml|use[- ]?case|"
    r"figure|image|screenshot|table|chart|graph|plot|visual|shown|"
    r"trend|axis|legend|arrow|actor|layout)\b",
    re.IGNORECASE,
)
EMBEDDING_DIMENSION = 384


def _tokens(text):
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


class OfflineEmbeddings(Embeddings):
    """Deterministic local fallback used when the Hugging Face download fails."""

    def __call__(self, text):
        # Older LangChain FAISS versions treat non-Embeddings objects as a
        # callable for query vectors.
        return self.embed_query(text)

    def _embed(self, text):
        vector = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
        for token in _tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSION
            vector[index] += 1.0
        norm = np.linalg.norm(vector)
        return (vector / norm if norm else vector).tolist()

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


def get_embeddings():
    """Prefer semantic embeddings, but keep indexing usable offline."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        ), "sentence-transformers/all-MiniLM-L6-v2"
    except Exception:
        return OfflineEmbeddings(), "offline-hash-embeddings"


class _HybridRetriever:
    def __init__(self, store, k=5, fetch_k=20):
        self.store = store
        self.k = k
        self.fetch_k = fetch_k

    def invoke(self, query):
        return self.store.hybrid_search(query, k=self.k, fetch_k=self.fetch_k)


class HybridVectorStore:
    """FAISS-backed store with lexical retrieval and rank fusion."""

    def __init__(self, faiss_store, documents):
        self._faiss = faiss_store
        self._documents = list(documents)
        self._postings = defaultdict(set)
        self._document_tokens = []
        for index, document in enumerate(self._documents):
            tokens = _tokens(document.page_content)
            self._document_tokens.append(Counter(tokens))
            for token in set(tokens):
                self._postings[token].add(index)

        # Keep this compatibility surface for the starter-question generator.
        self.docstore = self._faiss.docstore

    @property
    def documents(self):
        """Return indexed documents for diagnostics and multimodal context."""
        return list(self._documents)

    @property
    def visual_documents(self):
        """Return visual assets without changing the existing docstore API."""
        return [
            document for document in self._documents
            if document.metadata.get("is_visual")
        ]

    @property
    def has_visual_documents(self):
        return bool(self.visual_documents)

    def similarity_search(self, query, k=4):
        return self.hybrid_search(query, k=k, fetch_k=max(12, k * 3))

    def similarity_search_with_score(self, query, k=4):
        return self._faiss.similarity_search_with_score(query, k=k)

    def as_retriever(self, search_type="hybrid", search_kwargs=None):
        search_kwargs = search_kwargs or {}
        return _HybridRetriever(
            self,
            k=int(search_kwargs.get("k", 5)),
            fetch_k=int(search_kwargs.get("fetch_k", 20)),
        )

    def _lexical_search(self, query, limit):
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        query_counts = Counter(query_tokens)
        scores = Counter()
        for token in query_counts:
            for index in self._postings.get(token, ()):
                # Log-scaled term frequency rewards exact matches without
                # allowing repeated words to dominate the ranking.
                scores[index] += min(self._document_tokens[index][token], 3) * query_counts[token]
        return [index for index, _ in scores.most_common(limit)]

    def hybrid_search(self, query, k=5, fetch_k=20):
        dense = self._faiss.similarity_search(query, k=min(fetch_k, len(self._documents)))
        lexical_indexes = self._lexical_search(query, min(fetch_k, len(self._documents)))
        index_by_id = {id(document): index for index, document in enumerate(self._documents)}

        # The FAISS result objects are the same objects stored in docstore in
        # current LangChain versions; fall back to metadata/content matching
        # for versions that materialize copies.
        dense_indexes = []
        for document in dense:
            index = index_by_id.get(id(document))
            if index is None:
                index = next(
                    (
                        i for i, candidate in enumerate(self._documents)
                        if candidate.page_content == document.page_content
                        and candidate.metadata == document.metadata
                    ),
                    None,
                )
            if index is not None:
                dense_indexes.append(index)

        fused = {}
        for rank, index in enumerate(dense_indexes, start=1):
            fused[index] = fused.get(index, 0.0) + 1 / (60 + rank)
        for rank, index in enumerate(lexical_indexes, start=1):
            fused[index] = fused.get(index, 0.0) + 1 / (60 + rank)

        # Visual questions need the rendered asset/description as well as
        # nearby prose. Give visual records a small intent-aware boost while
        # retaining dense and lexical evidence for normal text questions.
        if _VISUAL_QUERY_RE.search(query or ""):
            for index in list(fused):
                metadata = self._documents[index].metadata
                if not metadata.get("is_visual"):
                    continue
                fused[index] += 0.035
                content_type = metadata.get("content_type")
                query_lower = (query or "").lower()
                if content_type == "diagram" and re.search(
                    r"\b(diagram|flowchart|workflow|architecture|uml|use[- ]?case|arrow)\b",
                    query_lower,
                ):
                    fused[index] += 0.065
                elif content_type == "table" and re.search(
                    r"\b(table|row|column|value)\b", query_lower
                ):
                    fused[index] += 0.065
                elif content_type == "chart" and re.search(
                    r"\b(chart|graph|trend|axis|plot)\b", query_lower
                ):
                    fused[index] += 0.065

        ranked = sorted(fused, key=fused.get, reverse=True)
        return [self._documents[index] for index in ranked[:k]]


def load_database(path="faiss_index"):
    embeddings, _ = get_embeddings()
    faiss_store = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    documents = list(faiss_store.docstore._dict.values())
    return HybridVectorStore(faiss_store, documents)
