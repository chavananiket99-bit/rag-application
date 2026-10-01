import fitz
import logging
import os
import threading
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

from rapidocr_onnxruntime import RapidOCR

logger = logging.getLogger("uvicorn.error")

# Keep OCR concurrency global so 100-file uploads cannot create one OCR
# engine per document/page. This is intentionally bounded and configurable.
MAX_PARALLEL_OCR = max(
    1,
    min(
        int(os.getenv("RAG_MAX_PARALLEL_OCR", "2")),
        (os.cpu_count() or 2),
    ),
)

OCR_EXECUTOR = ThreadPoolExecutor(
    max_workers=MAX_PARALLEL_OCR,
    thread_name_prefix="ocr-page",
)

# RapidOCR engines are kept thread-local. Each OCR worker owns one engine.
_engine_local = threading.local()


def _get_ocr_engine():
    engine = getattr(_engine_local, "engine", None)
    if engine is None:
        logger.info(
            "Initializing RapidOCR engine for thread: %s",
            threading.current_thread().name,
        )
        engine = RapidOCR()
        _engine_local.engine = engine
    return engine


def _ocr_page(pdf_path, page_number, dpi=150):
    """OCR one zero-based PDF page. Opens the PDF inside the worker."""
    doc = None
    temp_path = None

    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number)

        pix = page.get_pixmap(dpi=dpi)
        # Use an in-memory PNG instead of creating a persistent temporary
        # file for every page.
        image_bytes = pix.tobytes("png")
        image = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise RuntimeError("Unable to decode rendered PDF page for OCR.")
        result, _ = _get_ocr_engine()(image)

        parts = []
        if result:
            for item in result:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    detected_text = str(item[1]).strip()
                    if detected_text:
                        parts.append(detected_text)

        return " ".join(parts).strip()

    finally:
        if doc is not None:
            doc.close()


def read_scanned_pdf(
    pdf_path,
    page_numbers=None,
    progress_callback=None,
    total_pages=None,
):
    """
    OCR selected pages in parallel while preserving document page order.

    page_numbers is a list of zero-based page indexes. If omitted, every
    page is OCR'd. The caller owns the decision about which pages actually
    need OCR.
    """
    doc = fitz.open(pdf_path)
    try:
        document_page_count = len(doc)
    finally:
        doc.close()

    if page_numbers is None:
        page_numbers = list(range(document_page_count))
    else:
        page_numbers = sorted(set(int(p) for p in page_numbers))

    if total_pages is None:
        total_pages = document_page_count

    logger.info(
        "OCR started | file=%s | pages_to_ocr=%s | total_pages=%s | workers=%s",
        pdf_path,
        len(page_numbers),
        total_pages,
        MAX_PARALLEL_OCR,
    )

    results = {}
    completed = 0

    if not page_numbers:
        return results

    futures = {
        OCR_EXECUTOR.submit(_ocr_page, pdf_path, page_number): page_number
        for page_number in page_numbers
    }

    for future in futures:
        page_number = futures[future]
        try:
            results[page_number] = future.result()
        except Exception:
            logger.exception(
                "OCR failed for page %s",
                page_number + 1,
            )
            results[page_number] = ""

        completed += 1
        if progress_callback:
            try:
                progress_callback(completed, len(page_numbers), total_pages)
            except Exception:
                logger.debug(
                    "OCR progress callback failed.",
                    exc_info=True,
                )

    logger.info(
        "OCR completed | file=%s | pages=%s",
        pdf_path,
        len(page_numbers),
    )

    return results
