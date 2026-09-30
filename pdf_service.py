import logging
from services.pdf_reader import read_pdf
from services.ocr_reader import read_scanned_pdf
from services.chunk_service import split_text

logger = logging.getLogger("uvicorn.error")

MIN_USABLE_TEXT_LENGTH = 20

def _has_usable_text(pages):
    """
    Determine whether normal PDF extraction produced meaningful text.
    Some PDFs technically return a few characters, whitespace,
    page numbers, or garbage text even though OCR is required.
    """
    if not pages:
        return False

    extracted_text = "\n".join(
        str(page.get("text", "") or "")
        for page in pages
    ).strip()

    if len(extracted_text) < MIN_USABLE_TEXT_LENGTH:
        return False

    # Remove whitespace and check whether we have actual content.
    compact_text = "".join(
        extracted_text.split()
    )

    return len(compact_text) >= MIN_USABLE_TEXT_LENGTH

def _build_full_text(pages):
    """
    Build full document text while preserving page order.
    """
    return "\n".join(
        str(page.get("text", "") or "").strip()
        for page in pages
        if page.get("text")
    ).strip()

def extract_text(pdf_path):
    """
    Extract text from a PDF.
    Processing flow:
        PDF
            |
            +--> Normal text extraction
            |
            +--> Usable text?
                    |
                +----+----+
                |         |
                YES       NO
                |         |
                |        OCR
                |         |
                +----+----+
                    |  
                Page-aware
                chunks
    
    Returns:
        {
            "text": str,
            "pages": list[dict],
            "chunks": list[dict],
            "ocr_used": bool
        }
    """

    logger.info(
        "Extracting text from PDF: %s",
        pdf_path
    )

    # ============================================================
    # STEP 1 — NORMAL TEXT EXTRACTION
    # ============================================================
    try:
        pages = read_pdf(pdf_path)

    except Exception as exc:
        logger.exception(
            "Normal PDF text extraction failed: %s",
            exc
        )

        pages = []

    if pages is None:
        pages = []

    # ============================================================
    # STEP 2 — CHECK WHETHER NORMAL EXTRACTION IS USABLE
    # ============================================================
    ocr_used = False

    if not _has_usable_text(pages):
        logger.info(
            "PDF does not contain sufficient usable text. "
            "Performing OCR: %s",
            pdf_path
        )

        try:
            pages = read_scanned_pdf(pdf_path)
            ocr_used = True

        except Exception as exc:
            logger.exception(
                "OCR extraction failed: %s",
                exc
            )

            raise RuntimeError(
                "Unable to extract text from PDF."
            ) from exc

    if pages is None:
        pages = []

    # ============================================================
    # STEP 3 — BUILD FULL TEXT
    # ============================================================
    text = _build_full_text(pages)

    # ============================================================
    # STEP 4 — CHUNKING
    # ============================================================
    chunks = []

    if pages:
        try:
            chunks = split_text(pages)

        except Exception as exc:
            logger.exception(
                "PDF chunking failed: %s",
                exc
            )

            raise RuntimeError(
                "Unable to create document chunks."
            ) from exc

    if chunks is None:
        chunks = []

    # ============================================================
    # STEP 5 — LOGGING
    # ============================================================
    logger.info(
        "PDF processing completed | "
        "Pages=%s | Characters=%s | Chunks=%s | OCR=%s",
        len(pages),
        len(text),
        len(chunks),
        ocr_used,
    )

    return {
        "text": text,
        "pages": pages,
        "chunks": chunks,
        "ocr_used": ocr_used,
    }
