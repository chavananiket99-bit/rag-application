import fitz
import logging
import os
import tempfile

from rapidocr_onnxruntime import RapidOCR

logger = logging.getLogger("uvicorn.error")

engine = RapidOCR()

def read_scanned_pdf(pdf_path):
    """
    Perform OCR page by page on a scanned PDF.
    
    Returns:
        list[dict]: [
            {
                "page_number": 1,
                "text": "..."
            },
            ...
        ]
    """

    pages = []

    doc = None

    try:
        doc = fitz.open(pdf_path)

        logger.info(
            "OCR started. Total pages: %s",
            len(doc)
        )

        for page_number in range(
            len(doc)
        ):
            page = doc.load_page(
                page_number
            )

            logger.info(
                "Starting OCR for page %s",
                page_number + 1
            )

            temp_path = None

            try:
                pix = page.get_pixmap(
                    dpi=150
                )

                with tempfile.NamedTemporaryFile(
                    suffix=".png",
                    delete=False
                ) as temp_file:
                    temp_path = temp_file.name

                pix.save(
                    temp_path
                )

                result, _ = engine(
                    temp_path
                )

                page_text_parts = []

                if result:
                    for item in result:
                        if (
                            isinstance(item, (list, tuple))
                            and len(item) >= 2
                        ):
                            detected_text = str(
                                item[1]
                            ).strip()

                            if detected_text:
                                page_text_parts.append(
                                    detected_text
                                )
                page_text = " ".join(
                    page_text_parts
                ).strip()

                logger.info(
                    "--- OCR Page %s ---",
                    page_number + 1
                )

                if page_text:
                    logger.info(
                        "%s",
                        page_text
                    )
                else:
                    logger.info(
                        "[No OCR text on this page]"
                    )

                pages.append(
                    {
                        "page_number": page_number + 1,
                        "text": page_text,
                    }
                )

            except Exception as exc:
                logger.exception(
                    "OCR failed for page %s: %s",
                    page_number + 1,
                    exc
                )
                # Preserve page information even when
                # OCR fails for an individual page.
                pages.append(
                    {
                        "page_number": page_number + 1,
                        "text": "",
                    }
                )

            finally:
                if temp_path and os.path.exists(
                    temp_path
                ):
                    try:
                        os.remove(
                            temp_path
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to remove temporary OCR file %s: %s",
                            temp_path,
                            exc
                        )
        logger.info(
            "OCR completed successfully."
        )

        return pages

    finally:
        if doc is not None:
            doc.close()
