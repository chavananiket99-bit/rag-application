import logging

from services.pdf_reader import read_pdf
from services.ocr_reader import read_scanned_pdf
from services.chunk_service import split_text

logger = logging.getLogger("uvicorn.error")

MIN_USABLE_TEXT_LENGTH = 20


def _has_usable_page_text(text):
    if text is None:
        return False
    compact = "".join(str(text).split())
    return len(compact) >= MIN_USABLE_TEXT_LENGTH


def _build_full_text(pages):
    return "\n".join(
        str(page.get("text", "") or "").strip()
        for page in pages
        if page.get("text")
    ).strip()


def extract_text(pdf_path, progress_callback=None):
    """
    Page-aware extraction pipeline:
      1. PyMuPDF/pypdf text extraction first.
      2. Pages with sufficient native text bypass OCR.
      3. Only pages without usable text go to the bounded RapidOCR pool.
      4. OCR results are restored to original page order.
      5. Chunking happens after the complete ordered page list exists.
    """
    logger.info("Extracting text from PDF: %s", pdf_path)

    try:
        pages = read_pdf(pdf_path) or []
    except Exception as exc:
        logger.exception("Normal PDF text extraction failed: %s", exc)
        pages = []

    total_pages = len(pages)

    # If the normal reader could not even establish page structure, fall
    # back to OCR of the complete document.
    if not pages:
        logger.info("No native pages found. OCRing complete PDF: %s", pdf_path)
        ocr_results = read_scanned_pdf(
            pdf_path,
            progress_callback=progress_callback,
            total_pages=0,
        )
        doc_page_count = max(ocr_results.keys(), default=-1) + 1
        pages = [
            {"page_number": i + 1, "text": ocr_results.get(i, "")}
            for i in range(doc_page_count)
        ]
        ocr_used = bool(pages)
    else:
        ocr_page_numbers = [
            i for i, page in enumerate(pages)
            if not _has_usable_page_text(page.get("text", ""))
        ]

        native_count = total_pages - len(ocr_page_numbers)
        logger.info(
            "Page-aware extraction | total=%s | native=%s | OCR=%s",
            total_pages,
            native_count,
            len(ocr_page_numbers),
        )

        # Report native pages immediately so the UI reflects actual work.
        if progress_callback and total_pages:
            for i, page in enumerate(pages):
                if i not in set(ocr_page_numbers):
                    try:
                        progress_callback(i + 1, total_pages)
                    except Exception:
                        logger.debug("Progress callback failed.", exc_info=True)

        ocr_results = {}
        if ocr_page_numbers:
            def ocr_progress(completed_ocr, total_ocr, overall_total):
                # Progress is approximate here because native pages may have
                # already completed; the final ordered result is authoritative.
                completed_overall = min(
                    overall_total,
                    native_count + completed_ocr,
                )
                if progress_callback:
                    progress_callback(
                        completed_overall,
                        overall_total,
                    )

            ocr_results = read_scanned_pdf(
                pdf_path,
                page_numbers=ocr_page_numbers,
                progress_callback=ocr_progress,
                total_pages=total_pages,
            )

            for page_index, text in ocr_results.items():
                pages[page_index]["text"] = text

        ocr_used = bool(ocr_page_numbers)

    text = _build_full_text(pages)

    try:
        chunks = split_text(pages) if pages else []
    except Exception as exc:
        logger.exception("PDF chunking failed: %s", exc)
        raise RuntimeError("Unable to create document chunks.") from exc

    chunks = chunks or []

    logger.info(
        "PDF processing completed | Pages=%s | Characters=%s | Chunks=%s | OCR=%s",
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
