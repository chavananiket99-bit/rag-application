import logging
logger = logging.getLogger("uvicorn.error")

def split_text(
    pages,
    chunk_size=1000,
    overlap=200,
):
    """
    Split page-aware document text into overlapping chunks.
    If page_number is missing/None, the position of the page
    in the pages list is used as a fallback.
    """
    if not pages:
        return []

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size."
        )

    chunks = []

    for page_index, page in enumerate(
        pages,
        start=1
    ):
        if not isinstance(page, dict):
            continue

        # ----------------------------------------------------
        # Preserve supplied page number.
        # If missing/None, fall back to actual page position.
        # ----------------------------------------------------
        page_number = page.get(
            "page_number"
        )

        if page_number is None:
            page_number = page_index

        text = str(
            page.get(
                "text",
                ""
            )
        ).strip()

        if not text:
            continue

        text_length = len(text)

        start = 0

        while start < text_length:
            end = min(
                start + chunk_size,
                text_length
            )

            chunk_text = text[
                start:end
            ].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "page_number": page_number,
                    }
                )

            # Prevent infinite loops.
            if end >= text_length:
                break

            start += (
                chunk_size - overlap
            )

    logger.info(
        "Total chunks created: %s",
        len(chunks)
    )

    for index, chunk in enumerate(
        chunks
    ):
        logger.info(
            "--- Chunk %s | Page: %s | Characters: %s ---",
            index + 1,
            chunk.get("page_number"),
            len(
                chunk.get(
                    "text",
                    ""
                )
            ),
        )
    return chunks
