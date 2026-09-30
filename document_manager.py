import json
import logging
import uuid
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("uvicorn.error")

BASE_DIR = Path(__file__).resolve().parent.parent

METADATA_DIR = BASE_DIR / "metadata"
METADATA_FILE = METADATA_DIR / "documents.json"

class DocumentManager:
    def __init__(self):
        METADATA_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        if not METADATA_FILE.exists():
            self._atomic_write([])

    # ============================================================
    # INTERNAL WRITE
    # ============================================================
    def _atomic_write(self, documents):
        temp_file = (
            METADATA_FILE.with_suffix(".tmp")
        )

        try:
            with open(
                temp_file,
                "w",
                encoding="utf-8"
            ) as file:
                json.dump(
                    documents,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

                file.flush()

            temp_file.replace(
                METADATA_FILE
            )

        except Exception:
            if temp_file.exists():
                temp_file.unlink(
                    missing_ok=True
                )
            raise

    # ============================================================
    # GET DOCUMENTS
    # ============================================================
    def get_documents(self):
        if not METADATA_FILE.exists():
            return []

        try:
            with open(
                METADATA_FILE,
                "r",
                encoding="utf-8"
            ) as file:
                data = json.load(file)

            if not isinstance(data, list):
                logger.warning(
                    "documents.json does not contain a list. "
                    "Returning empty document list."
                )
                return []
            return data

        except json.JSONDecodeError as exc:
            logger.exception(
                "documents.json is corrupted: %s",
                exc
            )

            raise RuntimeError(
                "Document metadata is corrupted."
            ) from exc

    # ============================================================
    # SAVE DOCUMENTS
    # ============================================================
    def save_documents(self, documents):
        if not isinstance(documents, list):
            raise ValueError(
                "documents must be a list."
            )
        self._atomic_write(
            documents
        )

    # ============================================================
    # ADD DOCUMENT
    # ============================================================
    def add_document(
        self,
        filename,
        filesize,
        stored_filename=None,
        user_id=None,
        agreement_type=None,
        expiry_date=None,
        analysis=None,
    ):
        documents = self.get_documents()
        document = {
            "id": str(uuid.uuid4()),
            "filename": filename,
            "stored_filename": (
                stored_filename or filename
            ),
            "filesize": filesize,
            "uploaded_at": (
                datetime.now().isoformat()
            ),
            "status": "Indexed",
            "user_id": user_id,
            "agreement_type": agreement_type,
            "expiry_date": expiry_date,
            "analysis": analysis,
        }

        documents.append(
            document
        )
        self.save_documents(
            documents
        )

        logger.info(
            "Document metadata created | "
            "document_id=%s | filename=%s | user_id=%s",
            document["id"],
            filename,
            user_id
        )
        return document

    # ============================================================
    # DELETE DOCUMENT
    # ============================================================
    def delete_document(
        self,
        document_id
    ):
        if not document_id:
            return None

        documents = self.get_documents()
        document_to_delete = None

        for document in documents:
            if document.get("id") == document_id:
                document_to_delete = document
                break

        if document_to_delete is None:
            logger.info(
                "Document metadata not found: %s",
                document_id
            )
            return None

        remaining_documents = [
            document
            for document in documents
            if document.get("id") != document_id
        ]

        self.save_documents(
            remaining_documents
        )

        logger.info(
            "Document metadata deleted | "
            "document_id=%s",
            document_id
        )
        return document_to_delete

document_manager = DocumentManager()
