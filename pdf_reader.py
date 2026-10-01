from pypdf import PdfReader
import logging
logger = logging.getLogger("uvicorn.error")

def read_pdf(file_path):
    """
    Extract text from a text-based PDF page by page.
    Returns:
        list[dict]: [
            {
                "page_number": 1,
                "text": "..."
            },
            ...
        ]
    """
    reader = PdfReader(file_path)

    total_pages = len(reader.pages)

    logger.info(
        "Total pages in PDF: %s",
        total_pages
    )

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):
        try:
            page_text = page.extract_text() or ""

        except Exception as exc:
            logger.exception(
                "Failed to extract text from page %s: %s",
                page_number,
                exc
            )
            page_text = ""

        page_text = page_text.strip()

        logger.info(
            "--- Page %s ---",
            page_number
        )

        if page_text:
            logger.info(
                "%s",
                page_text
            )
        else:
            logger.info(
                "[No extractable text on this page]"
            )
        pages.append(
            {
                "page_number": page_number,
                "text": page_text,
            }
        )
    return pages
