def chunk_text(
        text: str,
        chunk_size: int = 1000,
        overlap: int = 150
) -> list[str]:

    """Split text into overlapping character chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk_Size")

    words = text.split()
    chunks = []

    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            overlap_words = []
            overlap_length = 0

            for previous_word in reversed(current_chunk):
                if overlap_length + len(previous_word) + 1 > overlap:
                    break
                overlap_words.insert(0, previous_word)
                overlap_length += len(previous_word) + 1


            current_chunk = overlap_words
            current_length = overlap_length

        current_chunk.append(word)
        current_length += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))


    return chunks