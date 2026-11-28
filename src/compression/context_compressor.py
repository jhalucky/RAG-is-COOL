
import re
from collections import Counter
from typing import Any


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then",
    "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "for", "from", "by", "with",
    "as", "at", "this", "that", "these", "those", "it",
    "its", "into", "about", "than", "their", "they",
    "them", "we", "our", "you", "your", "which", "what",
    "when", "where", "who", "how", "why", "can", "could",
    "would", "should", "may", "might", "will", "also",
}


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric terms."""
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def _content_tokens(text: str) -> set[str]:
    """Return meaningful tokens, excluding common stopwords."""
    return {
        token
        for token in _tokenize(text)
        if token not in STOPWORDS and len(token) > 1
    }


def _split_sentences(text: str) -> list[str]:
    """Split passage text into sentences."""
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]


def _count_tokens(text: str) -> int:
    """Approximate token count using word and punctuation units."""
    return len(re.findall(r"\w+|[^\w\s]", text))


def _sentence_score(
    sentence: str,
    query_tokens: set[str],
) -> float:
    """
    Score sentence relevance using lexical query overlap.

    This is a lightweight heuristic, not a semantic similarity score.
    """
    sentence_tokens = _content_tokens(sentence)

    if not sentence_tokens or not query_tokens:
        return 0.0

    overlap = sentence_tokens.intersection(query_tokens)

    return len(overlap) / (len(sentence_tokens) ** 0.5)


def compress_context(
    question: str,
    chunks: list[dict[str, Any]],
    max_tokens: int = 1800,
) -> dict[str, Any]:
    """
    Compress retrieved passages using query-focused sentence selection.

    Original chunk metadata is preserved. Only the text field is reduced.

    Returns compressed chunks and compression statistics.
    """

    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1.")

    if not chunks:
        return {
            "chunks": [],
            "original_tokens": 0,
            "compressed_tokens": 0,
            "tokens_saved": 0,
            "compression_ratio": 0.0,
        }

    query_tokens = _content_tokens(question)

    original_tokens = sum(
        _count_tokens(str(chunk.get("text", "")))
        for chunk in chunks
    )

    candidates = []
    seen_sentences = set()

    for chunk_index, chunk in enumerate(chunks):
        text = str(chunk.get("text", ""))

        for sentence_index, sentence in enumerate(
            _split_sentences(text)
        ):
            normalized = " ".join(_tokenize(sentence))

            if not normalized or normalized in seen_sentences:
                continue

            seen_sentences.add(normalized)

            candidates.append({
                "chunk_index": chunk_index,
                "sentence_index": sentence_index,
                "text": sentence,
                "score": _sentence_score(sentence, query_tokens),
                "tokens": _count_tokens(sentence),
            })

    # Select the most relevant sentences first.
    ranked = sorted(
        candidates,
        key=lambda item: (
            item["score"],
            -item["sentence_index"],
        ),
        reverse=True,
    )

    selected = []
    used_tokens = 0

    for candidate in ranked:
        if used_tokens + candidate["tokens"] > max_tokens:
            continue

        selected.append(candidate)
        used_tokens += candidate["tokens"]

    # If no sentence matched or fitted, retain the first sentence
    # that fits the budget instead of returning empty context.
    if not selected:
        for candidate in candidates:
            if candidate["tokens"] <= max_tokens:
                selected.append(candidate)
                used_tokens += candidate["tokens"]
                break

    # Reconstruct chunks in their original order.
    selected_by_chunk = {}

    for item in selected:
        selected_by_chunk.setdefault(
            item["chunk_index"], []
        ).append(item)

    compressed_chunks = []

    for chunk_index, chunk in enumerate(chunks):
        selected_sentences = selected_by_chunk.get(chunk_index)

        if not selected_sentences:
            continue

        selected_sentences.sort(
            key=lambda item: item["sentence_index"]
        )

        compressed_chunk = dict(chunk)
        compressed_chunk["text"] = " ".join(
            item["text"] for item in selected_sentences
        )

        compressed_chunks.append(compressed_chunk)

    compressed_tokens = sum(
        _count_tokens(chunk["text"])
        for chunk in compressed_chunks
    )

    tokens_saved = max(0, original_tokens - compressed_tokens)

    compression_ratio = (
        tokens_saved / original_tokens
        if original_tokens
        else 0.0
    )

    return {
        "chunks": compressed_chunks,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "tokens_saved": tokens_saved,
        "compression_ratio": round(compression_ratio, 4),
    }
