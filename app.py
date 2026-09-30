from fastapi.openapi.utils import get_openapi
from datetime import datetime, date, timedelta

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    Request,
    HTTPException,
    Depends,
)

from typing import Annotated

from fastapi.middleware.cors import CORSMiddleware

import shutil
import os
import time
import logging
import uuid
import re
import json

from pathlib import Path

from services.pdf_service import extract_text
from services.agreement_expiry_service import extract_agreement_expiry
from services.agreement_analysis_service import analyze_agreement
from models.question import QuestionRequest
from services.vector_store import vector_store
from services.gemini_service import ( ask_gemini, rewrite_followup_question )
from services.document_manager import document_manager
from services.auth_dependency import get_current_user
from services.conversation_service import conversation_service

from routers.auth import router as auth_router


# ============================================================
# APP
# ============================================================

app = FastAPI()

app.include_router(
    auth_router
)

logger = logging.getLogger(
    "uvicorn.error"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "https://apc01.safelinks.protection.outlook.com/?url=http%3A%2F%2F127.0.0.1%3A5173%2F&data=05%7C02%7Cchavan.aniket%40mahindra.com%7C1d21484b21bf4136feb308df07f669c9%7C8c4858b5f020483ab7ef71ded6e81767%7C0%7C0%7C639238427074011872%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&sdata=0jeosLYJzix%2F4anrrZfTOlG34fqHgi%2BF41v2wjXPPHU%3D&reserved=0",
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent

UPLOAD_FOLDER = (
    BASE_DIR / "uploads"
)

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# CONVERSATION FOLLOW-UP HELPERS
# ============================================================
def _normalize_document_name(filename):
    """
    Normalize a document filename for comparison.
    """
    if not filename:
       return ""

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        Path(filename).stem.lower(),
    ).strip()

def _document_reference_score(
    question,
    filename,
):
    """
    Determine whether the question explicitly refers
    to a particular uploaded document.
    This is deterministic and does not use Gemini.
    """
    if not question or not filename:
        return 0.0

    question_text = re.sub(
        r"[^a-z0-9]+",
        " ",
        question.lower(),
    ).strip()

    filename_text = _normalize_document_name(
        filename
    )

    if not question_text or not filename_text:
        return 0.0

    question_tokens = set(
        question_text.split()
    )

    filename_tokens = set(
        filename_text.split()
    )

    if not question_tokens or not filename_tokens:
        return 0.0

    # --------------------------------------------------------
    # Common aliases
    # --------------------------------------------------------
    aliases = {
        "wh": "warehouse",
        "whs": "warehouse",
        "warehouses": "warehouse",
        "off": "office",
        "offs": "office",
        "offices": "office",
        "out": "outbound",
        "in": "inbound",
    }

    question_tokens = {
        aliases.get(
            token,
            token
        )
        for token in question_tokens
    }

    filename_tokens = {
        aliases.get(
            token,
            token
        )
        for token in filename_tokens
    }

    # --------------------------------------------------------
    # Ignore generic words
    # --------------------------------------------------------
    generic_terms = {
        "what",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "of",
        "for",
        "to",
        "in",
        "on",
        "at",
        "and",
        "or",
        "with",
        "from",
        "by",
        "how",
        "much",
        "about",
        "this",
        "that",
        "one",
        "document",
        "file",
        "pdf",
        "same",
        "also",
        "please",
        "tell",
        "me",
    }

    question_tokens -= generic_terms
    filename_tokens -= generic_terms

    if not question_tokens or not filename_tokens:
        return 0.0

    overlap = (
        question_tokens
        .intersection(
            filename_tokens
        )
    )
    if not overlap:
        return 0.0

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------
    query_coverage = (
        len(overlap)
        / len(question_tokens)
    )

    filename_coverage = (
        len(overlap)
        / len(filename_tokens)
    )

    score = (
        query_coverage * 0.60
        + filename_coverage * 0.40
    )

    return min(
        1.0,
        float(score)
    )

def _detect_explicit_document(
    question,
    available_documents,
):
    """
    Find an explicitly referenced document.
    Returns:
        filename or None
    """

    if not question:
        return None

    if not available_documents:
        return None

    # ========================================================
    # EXACT DOCUMENT MATCH
    # ========================================================
    normalized_question = _normalize_document_name(
        question
    )

    if normalized_question:
        for filename in available_documents:
            normalized_filename = (
                _normalize_document_name(
                    filename
                )
            )

            if not normalized_filename:
                continue

            # Exact filename/stem match
            if (
                normalized_question
                == normalized_filename
            ):
                logger.info(
                    "Exact explicit document detected | "
                    "question=%s | document=%s",
                    question,
                    filename,
                )
                return filename

    scored_documents = []

    for filename in available_documents:
        score = _document_reference_score(
            question,
            filename,
        )

        if score <= 0:
            continue

        scored_documents.append(
            (
                score,
                filename,
            )
        )

    if not scored_documents:
        return None

    scored_documents.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    best_score, best_filename = (
        scored_documents[0]
    )

    # Require meaningful document overlap.
    if best_score < 0.40:
        return None

    # Avoid ambiguous matches.
    if len(scored_documents) > 1:
        second_score = (
            scored_documents[1][0]
        )

        if (
            abs(
                best_score
                - second_score
            )
            < 0.10
        ):
            return None

    logger.info(
        "Explicit document detected | "
        "question=%s | document=%s | score=%.3f",
        question,
        best_filename,
        best_score,
    )

    return best_filename

def _get_document_resolution_message(
    resolution
):
    candidates = resolution.get(
        "candidates",
        []
    )

    if not candidates:
        return None
    
    names = [
        item.get("filename")
        for item in candidates
        if item.get("filename")
    ]

    if not names:
        return None

    return (
        "I found multiple documents that may match your "
        "question. Please specify the document name. "
        "Possible matches: "
        + ", ".join(names[:5])
    )

def _detect_follow_up_question(
    question,
    last_question=None,
    last_answer=None,
    current_documents=None,
):
    """
    Conservative follow-up detection.
    Returns True only when the new question appears to
    depend on the previous conversation.
    """
    if not question:
        return False

    normalized = re.sub(
        r"\s+",
        " ",
        question.lower(),
    ).strip()

    if not normalized:
        return False

    # --------------------------------------------------------
    # Explicit follow-up phrases
    # --------------------------------------------------------
    follow_up_patterns = [
        r"\bwhat about\b",
        r"\bhow about\b",
        r"\band the\b",
        r"\band for\b",
        r"\band what about\b",
        r"\bwhat of\b",
        r"\bthe other one\b",
        r"\bthe other document\b",
        r"\bthe second one\b",
        r"\bthe first one\b",
        r"\bthis one\b",
        r"\bthat one\b",
        r"\bsame for\b",
        r"\bsame as\b",
        r"\bwhat about its\b",
        r"\bwhat about their\b",
        r"\band the same\b",
    ]

    if any(
        re.search(
            pattern,
            normalized,
        )
        for pattern in follow_up_patterns
    ):
        return bool(
            last_question
            or last_answer
            or current_documents
        )

    # --------------------------------------------------------
    # Pronoun/reference-based short questions
    # --------------------------------------------------------
    reference_words = {
        "it",
        "its",
        "they",
        "their",
        "them",
        "that",
        "this",
        "one",
        "same",
        "other",
    }

    tokens = set(
        re.findall(
            r"\b[a-z0-9]+\b",
            normalized,
        )
    )

    if (
        tokens.intersection(
            reference_words
        )
        and (
            last_question
            or last_answer
            or current_documents
        )
    ):
        return True

    # --------------------------------------------------------
    # Very short questions following an existing question
    #
    # Examples:
    #
    # "And the rate?"
    # "And renewal?"
    # "The duration?"
    # --------------------------------------------------------
    question_tokens = re.findall(
        r"\b[a-z0-9]+\b",
        normalized,
    )

    if (
        last_question
        and len(question_tokens) <= 5
    ):
        follow_up_starters = {
            "and",
            "also",
            "then",
            "what",
            "how",
            "the",
            "about",
        }

        if (
            question_tokens
            and question_tokens[0]
            in follow_up_starters
        ):
            return True

    return False


# ============================================================
# REQUEST LOGGING
# ============================================================

@app.middleware("http")
async def log_requests(
    request: Request,
    call_next
):

    start_time = time.time()

    logger.info(
        "Request started: %s %s",
        request.method,
        request.url.path,
    )

    response = await call_next(
        request
    )

    duration_ms = (
        time.time()
        - start_time
    ) * 1000

    logger.info(
        "Request completed: %s %s -> %s (%.2f ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )

    return response


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message":
            "PDF RAG Backend Running"
    }


# ============================================================
# UPLOAD
# ============================================================

@app.post("/upload")
async def upload_files(
    files: Annotated[
        list[UploadFile],
        File(...)
    ],
    agreement_metadata: Annotated[
        str | None,
        Form()
    ] = None,
    current_user: dict = Depends(
        get_current_user
    ),
):

    logger.info(
        "Multiple upload request received. Files: %s",
        len(files)
    )

    uploaded_documents = []
    failed_documents = []

    user_id = current_user["id"]

    try:
        parsed_agreement_metadata = (
            json.loads(agreement_metadata)
            if agreement_metadata
            else []
        )
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="Agreement metadata must be valid JSON.",
        ) from exc

    if not isinstance(parsed_agreement_metadata, list):
        raise HTTPException(
            status_code=400,
            detail="Agreement metadata must be a list.",
        )

    for file_index, file in enumerate(files):

        metadata = (
            parsed_agreement_metadata[file_index]
            if file_index < len(parsed_agreement_metadata)
            and isinstance(parsed_agreement_metadata[file_index], dict)
            else {}
        )
        expiry_date = metadata.get("expiry_date") or None
        agreement_type = metadata.get("agreement_type") or None

        if expiry_date:
            try:
                date.fromisoformat(expiry_date)
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Agreement expiry date must use "
                        "YYYY-MM-DD format."
                    ),
                ) from exc

        original_filename = Path(
            file.filename
            or "uploaded_file.pdf"
        ).name

        stored_filename = None
        file_path = None
        document = None

        logger.info(
            "Processing uploaded file: %s",
            original_filename
        )

        try:

            # ====================================================
            # SAVE PHYSICAL FILE
            # ====================================================

            unique_id = (
                uuid.uuid4().hex
            )

            stored_filename = (
                f"{unique_id}_{original_filename}"
            )

            file_path = (
                UPLOAD_FOLDER
                / stored_filename
            )

            with open(
                file_path,
                "wb"
            ) as buffer:

                shutil.copyfileobj(
                    file.file,
                    buffer
                )

            file_size = (
                os.path.getsize(
                    file_path
                )
            )

            logger.info(
                "File saved: %s | bytes=%s",
                original_filename,
                file_size
            )

            # ====================================================
            # EXTRACT PDF
            # ====================================================

            result = extract_text(
                str(file_path)
            )

            text = result.get(
                "text",
                ""
            )

            pages = result.get(
                "pages",
                []
            )

            chunks = result.get(
                "chunks",
                []
            )

            ocr_used = result.get(
                "ocr_used",
                False
            )

            logger.info(
                "Text extraction completed: %s | "
                "Characters=%s | Pages=%s | "
                "Chunks=%s | OCR=%s",
                original_filename,
                len(text),
                len(pages),
                len(chunks),
                ocr_used,
            )

            if not chunks:

                raise ValueError(
                    "No text chunks could be extracted "
                    "from the PDF."
                )

            if not expiry_date:
                expiry_date = extract_agreement_expiry(
                    text
                )

            if expiry_date and not agreement_type:
                agreement_type = "Agreement"

            agreement_analysis = analyze_agreement(
                text,
                expiry_override=expiry_date,
            )

            if agreement_analysis.get("expiry_date"):
                expiry_date = agreement_analysis["expiry_date"]
                agreement_type = agreement_type or "Agreement"

            # ====================================================
            # CREATE DOCUMENT METADATA
            # ====================================================

            document = (
                document_manager.add_document(
                    filename=original_filename,
                    filesize=file_size,
                    stored_filename=stored_filename,
                    user_id=user_id,
                    agreement_type=agreement_type,
                    expiry_date=expiry_date,
                    analysis=agreement_analysis,
                )
            )

            # ====================================================
            # ADD TO FAISS
            # ====================================================

            faiss_chunks_added = (
                vector_store.add_document(
                    document_id=document["id"],
                    filename=original_filename,
                    chunks=chunks,
                    user_id=user_id,
                )
            )

            if (
                faiss_chunks_added
                != len(chunks)
            ):

                raise RuntimeError(
                    "FAISS indexing did not add "
                    "all document chunks."
                )

            logger.info(
                "FAISS indexing completed: %s | chunks=%s",
                original_filename,
                faiss_chunks_added
            )

            # ====================================================
            # SUCCESS
            # ====================================================

            uploaded_documents.append(
                {
                    "document":
                        document,

                    "characters":
                        len(text),

                    "pages":
                        len(pages),

                    "chunks":
                        len(chunks),

                    "faiss_chunks_added":
                        faiss_chunks_added,

                    "ocr_used":
                        ocr_used,

                    "preview":
                        text[:500],
                }
            )

        except Exception as exc:

            logger.exception(
                "Failed to process file: %s",
                original_filename
            )

            # ====================================================
            # ROLLBACK FAISS
            # ====================================================

            if document:

                try:

                    vector_store.delete_document(
                        document["id"]
                    )

                except Exception:

                    logger.exception(
                        "FAISS rollback failed for %s",
                        original_filename
                    )

                # ------------------------------------------------
                # Roll back document metadata if possible.
                # ------------------------------------------------

                try:

                    document_manager.delete_document(
                        document["id"]
                    )

                except Exception:

                    logger.exception(
                        "Document metadata rollback failed "
                        "for %s",
                        original_filename
                    )

            # ====================================================
            # ROLLBACK PHYSICAL FILE
            # ====================================================

            try:

                if (
                    file_path
                    and file_path.exists()
                ):

                    file_path.unlink()

                    logger.info(
                        "Rolled back physical file: %s",
                        file_path
                    )

            except Exception:

                logger.exception(
                    "Failed to remove incomplete file: %s",
                    original_filename
                )

            failed_documents.append(
                {
                    "filename":
                        original_filename,

                    "error":
                        str(exc),
                }
            )

    logger.info(
        "Multiple upload completed. "
        "Success=%s | Failed=%s",
        len(uploaded_documents),
        len(failed_documents)
    )

    return {

        "message":
            "Upload processing completed.",

        "total_files":
            len(files),

        "successful_files":
            len(uploaded_documents),

        "failed_files":
            len(failed_documents),

        "documents":
            uploaded_documents,

        "failures":
            failed_documents,
    }

def _detect_question_intent(question):
    """
    Deterministically classify the user's question.
    This is used only to guide retrieval.
    It does NOT answer the question.
    """
    if not question:
        return "general"

    normalized = re.sub(
        r"\s+",
        " ",
        question.lower()
    ).strip()

    signature_terms = [
        "signatory",
        "signatories",
        "signature",
        "signed",
        "signed by",
        "who signed",
        "who has signed",
        "authorized signatory",
        "authorised signatory",
        "esign",
        "e sign",
        "e-sign",
        "electronic signature",
        "electronically signed",
        "digitally signed",
        "digital signature",
        "signdesk",
        "execution",
    ]

    lifecycle_terms = [
        "effective date",
        "start date",
        "commencement date",
        "expiry",
        "expiry date",
        "end date",
        "termination date",
        "tenure",
        "term",
        "duration",
        "validity",
        "valid until",
        "active",
        "expired",
        "renewal",
        "renew",
        "agreement period",
    ]

    commercial_terms = [
        "price",
        "pricing",
        "rate",
        "rent",
        "rental",
        "cost",
        "amount",
        "charge",
        "payment",
        "payment terms",
        "commercial",
        "markup",
        "mark-up",
        "margin",
        "fee",
        "consideration",
    ]

    purpose_terms = [
        "purpose",
        "business purpose",
        "business rationale",
        "business need",
        "objective",
        "strategic objective",
        "why was",
        "why did",
        "reason for",
        "commercial intent",
        "business justification",
    ]

    transaction_terms = [
        "transaction",
        "nature of transaction",
        "nature of activity",
        "activity",
        "activities",
        "sale",
        "purchase",
        "service",
        "services",
        "design service",
        "manufacturing",
        "contract manufacturing",
        "royalty",
        "procurement",
    ]

    actualization_terms = [
        "actualization",
        "actualisation",
        "actualize",
        "actualise",
        "actualized",
        "actualised",
    ]

    if any(
        term in normalized
        for term in signature_terms
    ):
        return "signature"

    if any(
        term in normalized
        for term in lifecycle_terms
    ):
        return "lifecycle"

    if any(
        term in normalized
        for term in actualization_terms
    ):
        return "actualization"

    if any(
        term in normalized
        for term in commercial_terms
    ):
        return "commercial"

    if any(
        term in normalized
        for term in purpose_terms
    ):
        return "purpose"

    if any(
        term in normalized
        for term in transaction_terms
    ):
        return "transaction"

    return "general"


def _is_comparison_question(question):
    """Return whether the question explicitly compares agreements."""
    normalized = (question or "").lower()
    comparison_terms = [
        "compare",
        "comparison",
        "comparative",
        "versus",
        " vs ",
        "difference between",
        "differences between",
        "both agreements",
        "all agreements",
        "same clause",
        "which agreement",
    ]
    return any(
        term in normalized
        for term in comparison_terms
    )


def _answer_agreement_portfolio_question(question, documents):
    normalized = re.sub(r"\s+", " ", (question or "").lower()).strip()

    if "agreement" not in normalized:
        return None

    today = date.today()
    agreements = [
        document
        for document in documents
        if document.get("expiry_date")
    ]
    selected = None
    heading = None

    days_match = re.search(
        r"(?:expir\w*|due)\s+(?:in|within)\s+(?:the\s+)?next\s+(\d+)\s+days",
        normalized,
    )

    if days_match:
        days = int(days_match.group(1))
        deadline = today + timedelta(days=days)
        selected = [
            document
            for document in agreements
            if today <= date.fromisoformat(document["expiry_date"]) <= deadline
        ]
        heading = f"Agreements expiring in the next {days} days"
    elif re.search(r"\bexpired\s+agreements?\b", normalized):
        selected = [
            document
            for document in agreements
            if date.fromisoformat(document["expiry_date"]) < today
        ]
        heading = "Expired agreements"
    elif re.search(r"\bactive\s+agreements?\b", normalized):
        selected = [
            document
            for document in agreements
            if date.fromisoformat(document["expiry_date"]) >= today
        ]
        heading = "Active agreements"

    if selected is None:
        return None

    selected.sort(
        key=lambda document: document["expiry_date"]
    )

    if not selected:
        return f"{heading}: none found."

    lines = [f"{heading} ({len(selected)}):"]
    lines.extend(
        f"- {document.get('filename', 'Unnamed agreement')}: "
        f"{document['expiry_date']}"
        for document in selected
    )
    return "\n".join(lines)

def _is_strong_enough_result(
    result,
    intent
):
    """
    Keep a retrieved chunk only when it contains meaningful
    evidence for the question. Retrieval rank/metadata alone
    is never treated as factual evidence.
    """
    if not result:
        return False

    relevance = float(
        result.get(
            "relevance_score",
            0.0
        )
    )

    semantic = float(
        result.get(
            "semantic_score",
            0.0
        )
    )

    lexical = float(
        result.get(
            "lexical_score",
            0.0
        )
    )

    signature = float(
        result.get(
            "signature_score",
            0.0
        )
    )

    financial = float(
        result.get(
            "financial_score",
            0.0
        )
    )

    lifecycle = float(
        result.get(
            "lifecycle_score",
            0.0
        )
    )

    purpose = float(
        result.get(
            "purpose_score",
            0.0
        )
    )

    transaction = float(
        result.get(
            "transaction_score",
            0.0
        )
    )

    if intent == "signature":
        return (
            signature >= 0.20
            or relevance >= 0.20
            or (semantic >= 0.40 and lexical >= 0.25)
        )

    if intent == "commercial":
        return (
            financial >= 0.20
            or relevance >= 0.20
            or (semantic >= 0.40 and lexical >= 0.25)
        )

    if intent == "lifecycle":
        return (
            lifecycle >= 0.15
            or relevance >= 0.20
            or lexical >= 1.0
            or (semantic >= 0.40 and lexical >= 0.25)
        )

    if intent == "purpose":
        return (
            purpose >= 0.15
            or relevance >= 0.20
            or lexical >= 1.0
            or (semantic >= 0.40 and lexical >= 0.25)
        )

    if intent == "transaction":
        return (
            transaction >= 0.15
            or relevance >= 0.20
            or lexical >= 1.0
            or (semantic >= 0.40 and lexical >= 0.25)
        )

    if intent == "actualization":
        return (
            result.get("intent_score", 0.0) >= 0.25
            or relevance >= 0.20
            or lexical >= 1.0
            or (semantic >= 0.40 and lexical >= 0.25)
        )

    return (
        relevance >= 0.15
        or lexical >= 1.0
        or (semantic >= 0.40 and lexical >= 0.25)
    )

def _deduplicate_retrieved_results(results):
    """
    Remove duplicate retrieved evidence while preserving
    distinct chunks/pages from the same document.
    Two results are considered duplicates when they point to
    the same document, page and chunk.
    Retrieval score is preserved by keeping the strongest
    occurrence of each evidence location.
    """
    if not results:
       return []
   
    
    unique_results = {}
    
    for result in results:
        if not result:
            continue
        
        result_key = (
            result.get("document_id"),
            result.get("filename"),
            result.get("page_number"),
            result.get("chunk_index"),
        )
        
        existing = unique_results.get(
            result_key
        )
        
        if existing is None:
            unique_results[result_key] = result
            continue
        
        existing_score = float(
            existing.get(
                "final_score",
                0.0
            )
        )
        
        current_score = float(
            result.get(
                "final_score",
                0.0
            )
        )

        if current_score > existing_score:
            unique_results[result_key] = result
    
    deduplicated_results = list(
        unique_results.values()
    )
    
    # Keep strongest evidence first.
    deduplicated_results.sort(
        key=lambda item: float(
            item.get(
                "final_score",
                0.0
            )
        ),
        reverse=True,
    )
    
    logger.info(
        "Evidence deduplication | "
        "before=%s | after=%s | removed=%s",
        len(results),
        len(deduplicated_results),
        len(results) - len(deduplicated_results),
    )
    return deduplicated_results

# ============================================================
# ASK
# ============================================================

@app.post("/ask")
async def ask_question(
    request: QuestionRequest,
    current_user: dict = Depends(
        get_current_user
    ),
):
    question = (
        request.question.strip()
        if request.question
        else ""
    )

    user_id = current_user["id"]

    # ========================================================
    # VALIDATE QUESTION
    # ========================================================
    if not question:
        return {
            "question": question,
            "answer":
                "Please enter a question.",
            "sources": [],
            "conversation_id":
                request.conversation_id,
        }

    # ========================================================
    # GET CURRENT USER DOCUMENTS
    #
    # These are ALL documents currently uploaded by the user.
    # They are stored separately from current_documents,
    # which represents documents used by the latest answer.
    # ========================================================
    all_documents = (
        document_manager
        .get_documents()
    )

    _backfill_agreement_analysis(
        all_documents
    )

    user_documents = [
        document
        for document in all_documents
        if document.get(
            "user_id"
        ) == user_id
    ]

    available_document_names = []

    available_document_ids = []

    for document in user_documents:
        document_id = document.get(
            "id"
        )

        filename = document.get(
            "filename"
        )

        if (
            document_id
            and document_id
            not in available_document_ids
        ):
            available_document_ids.append(
                document_id
            )

        if (
            filename
            and filename
            not in available_document_names
        ):
            available_document_names.append(
                filename
            )

    logger.info(
        "Available user documents | "
        "user=%s | documents=%s",
        user_id,
        available_document_names,
    )

    # ========================================================
    # GET OR CREATE CONVERSATION
    # ========================================================
    conversation = (
        conversation_service
        .get_or_create_conversation(
            user_id=user_id,
            conversation_id=(
                request.conversation_id
            ),
            document_ids=(
                available_document_ids
            ),
            document_names=(
                available_document_names
            ),
        )
    )

    conversation_id = conversation[
        "id"
    ]

    # ========================================================
    # ALWAYS REFRESH AVAILABLE DOCUMENTS
    #
    # This is important because the user can upload another
    # PDF after a conversation has already started.
    #
    # Example:
    #
    # Q1 -> Inbound.pdf uploaded
    # Upload -> Outbound.pdf
    # Q2 -> "What about the outbound one?"
    #
    # The existing conversation must now know about both.
    # ========================================================
    conversation_service.set_available_documents(
        user_id=user_id,
        conversation_id=conversation_id,
        document_ids=available_document_ids,
        document_names=available_document_names,
    )

    portfolio_answer = _answer_agreement_portfolio_question(
        question,
        user_documents,
    )

    if portfolio_answer is not None:
        conversation_service.add_message(
            user_id=user_id,
            conversation_id=conversation_id,
            question=question,
            answer=portfolio_answer,
            rewritten_question=question,
            sources=[],
        )
        return {
            "question": question,
            "answer": portfolio_answer,
            "sources": [],
            "conversation_id": conversation_id,
        }

    logger.info(
        "Conversation available documents updated | "
        "conversation=%s | documents=%s",
        conversation_id,
        available_document_names,
    )

    logger.info(
        "Received question: %s | "
        "user=%s | conversation=%s",
        question,
        user_id,
        conversation_id,
    )

    # ========================================================
    # PREVIOUS CONVERSATION CONTEXT
    # ========================================================
    conversation_context = (
        conversation_service.get_context(
            user_id=user_id,
            conversation_id=conversation_id,
        )
    )

    last_question = (
        conversation_context.get(
            "last_question"
        )
    )

    last_answer = (
        conversation_context.get(
            "last_answer"
        )
    )

    current_documents = (
        conversation_context.get(
            "current_documents",
            []
        )
    )

    current_document = (
        conversation_context.get(
            "current_document"
        )
    )

    available_documents = (
        conversation_context.get(
            "available_documents",
            []
        )
    )

    logger.info(
        "Conversation context | "
        "conversation=%s | "
        "last_question=%s | "
        "last_answer=%s | "
        "current_documents=%s | "
        "available_documents=%s",
        conversation_id,
        last_question,
        last_answer,
        current_documents,
        available_documents,
    )

    # ========================================================
    # FOLLOWUP DETECTION
    # ========================================================
    is_follow_up = _detect_follow_up_question(
        question=question,
        last_question=last_question,
        last_answer=last_answer,
        current_documents=current_documents,
    )

    # ========================================================
    # EXPLICIT DOCUMENT REFERENCE
    # ========================================================
    explicit_document = _detect_explicit_document(
        question=question,
        available_documents=available_documents,
    )

    logger.info(
        "Question Context detection | "
        "conversation=%s | "
        "is_follow_up=%s | "
        "explicit_document=%s",
        conversation_id,
        is_follow_up,
        explicit_document,
    )

    # ========================================================
    # REWRITE FOLLOW-UP QUESTION
    # ========================================================
    rewritten_question = question
    target_document = explicit_document

    if is_follow_up:
        rewrite_result = rewrite_followup_question(
            question=question,
            last_question=last_question,
            last_answer=last_answer,
            current_documents=current_documents,
            explicit_document=explicit_document,
            current_document=current_document,
        )

        rewritten_question = (
            rewrite_result.get(
                "standalone_question",
                question,
            )
            or question
        )

        rewrite_target_document = (
            rewrite_result.get(
                "target_document"
            )
        )

        # ========================================================
        # DOCUMENT TARGET PRIORITY
        #
        # 1. Explicit document in current question
        # 2. Valid document returned by follow-up rewriter
        # 3. Current conversation document
        # ========================================================
        if explicit_document:
            target_document = explicit_document

        elif (
            rewrite_target_document
            and rewrite_target_document
            in available_documents
        ):
            target_document = rewrite_target_document

        elif current_document:
            target_document = current_document

        elif current_documents:
            target_document = current_documents[0]

        else:
            target_document = None

        logger.info(
            "Follow-up rewrite | "
            "conversation=%s | "
            "original=%s | "
            "rewritten=%s | "
            "target_document=%s",
            conversation_id,
            question,
            rewritten_question,
            target_document,
        )
    else:
        logger.info(
            "Question is standalone | question=%s",
            question,
        )

    # ========================================================
    # FINAL DOCUMENT RESOLUTION
    # ========================================================
    
    resolved_filename = target_document

    document_resolution = {
        "filename": resolved_filename,
        "score": 0.0,
        "ambiguous": False,
        "candidates": [],
    }

    is_comparison_question = _is_comparison_question(
        rewritten_question
    )

    comparison_documents = set()
    if is_comparison_question:
        comparison_documents = (
            vector_store._detect_target_documents(
                rewritten_question,
                user_id=user_id,
            )
        )

    if not resolved_filename:
        document_resolution = vector_store.resolve_document(
            rewritten_question,
            user_id=user_id,
        )

        resolved_filename = (
            document_resolution.get("filename")
        )

        if (
            document_resolution.get("ambiguous")
            and len(comparison_documents) < 2
        ):
            return {
                "question": question,
                "answer": _get_document_resolution_message(document_resolution),
                "sources": [],
                "conversation_id": conversation_id,
            }

    logger.info(
        "Final document resolution | "
        "conversation=%s | "
        "original_question=%s | "
        "rewritten_question=%s | "
        "target_document=%s | "
        "resolved_filename=%s",
        conversation_id,
        question,
        rewritten_question,
        target_document,
        resolved_filename,
    )

    normalized_question = (
        rewritten_question.lower()
    )

    signature_keywords = [
        "signatory",
        "signatories",
        "signature",
        "signatures",
        "signed",
        "signed by",
        "who signed",
        "who has signed",
        "who executed",
        "execution",
        "authorized signatory",
        "authorised signatory",
        "authorized representative",
        "authorised representative",
        "representative",
        "esign",
        "e sign",
        "e-sign",
        "electronic signature",
        "electronically signed",
        "digital signed",
        "digital signature",
        "signed electronically",
        "sign desk",
        "signdesk"
    ]

    is_signature_question = any(
        keyword
        in normalized_question
        for keyword
        in signature_keywords
    )

    # ========================================================
    # RETRIEVE
    # ========================================================
    # retrieval_k = (
    #     8
    #     if is_signature_question
    #     else 5
    # )

    # ========================================================
    # QUESTION INTENT + RETRIEVAL SIZE
    # ========================================================
    question_intent = _detect_question_intent(
        rewritten_question
    )

    retrieval_k_by_intent = {
        "signature": 10,
        "lifecycle": 8,
        "commercial": 8,
        "purpose": 7,
        "transaction": 7,
        "actualization": 8,
        "general": 5,
    }

    retrieval_k = retrieval_k_by_intent.get(
        question_intent,
        5
    )

    is_signature_question = (
        question_intent == "signature"
    )

    logger.info(
        "Question intent resolved | "
        "question=%s | intent=%s | retrieval_k=%s",
        rewritten_question,
        question_intent,
        retrieval_k,
    )

    # --------------------------------------------------------
    # CRITICAL DOCUMENT ISOLATION
    #
    # If a document has been explicitly identified, NEVER use
    # normal cross-document hybrid search.
    #
    # Search only inside the selected document.
    # This prevents a semantically similar chunk from another
    # PDF from entering the Gemini context.
    # --------------------------------------------------------
    if len(comparison_documents) >= 2:
        logger.info(
            "Comparison retrieval across documents | "
            "documents=%s | query=%s",
            sorted(comparison_documents),
            rewritten_question,
        )

        results = []
        for filename in sorted(comparison_documents):
            results.extend(
                vector_store.search_in_document(
                    query=rewritten_question,
                    filename=filename,
                    top_k=retrieval_k,
                    user_id=user_id,
                )
            )

        unique_results = {}
        for result in results:
            result_key = (
                result.get("filename"),
                result.get("chunk_index"),
                result.get("page_number"),
            )
            unique_results[result_key] = result

        results = sorted(
            unique_results.values(),
            key=lambda item: item.get(
                "final_score",
                0.0,
            ),
            reverse=True,
        )

    elif resolved_filename:
        logger.info(
            "Document-restricted retrieval | "
            "document=%s | query=%s",
            resolved_filename,
            rewritten_question,
        )

        results = vector_store.search_in_document(
            query=rewritten_question,
            filename=resolved_filename,
            top_k=retrieval_k,
            user_id=user_id,
        )
    else:
        logger.info(
            "Cross-document retrieval | query=%s",
            rewritten_question,
        )

        results = vector_store.search(
            query=rewritten_question,
            top_k=retrieval_k,
            user_id=user_id,
            resolved_filename=None,
        )
            
    # ========================================================
    # NO RESULTS
    # ========================================================
    if not results:
        if not available_document_names:
            answer = (
                "Please upload PDF first."
            )
        else:
            answer = (
                "I couldn't find that information "
                "in the uploaded document."
            )
            
        return {
            "question":
                question,

            "answer":
                answer,

            "sources":
                [],

            "conversation_id":
                conversation_id,
        }

    # ============================================================
    # FILTER WEAK RESULTS
    # ============================================================
    strong_results = [
        result
        for result in results
        if _is_strong_enough_result(
            result,
            question_intent
        )
    ]
    
    if not strong_results:
        logger.warning(
            "No sufficiently strong evidence found | "
            "intent=%s | question=%s",
            question_intent,
            rewritten_question,
        )
        
        return {
            "question": question,
            "answer":
                "I couldn't find that information "
                "in the uploaded document.",
            "sources": [],
            "conversation_id":
                conversation_id,
        }
    
    # ============================================================
    # DEDUPLICATE RETRIEVED EVIDENCE
    # ============================================================
    results = _deduplicate_retrieved_results(
        strong_results
    )
    
    if not results:
        logger.warning(
            "No evidence remained after deduplication | "
            "question=%s",
            rewritten_question,
        )
        
        return {
            "question": question,
            "answer":
                "I couldn't find that information "
                "in the uploaded document.",
            "sources": [],
            "conversation_id":
                conversation_id,
        }

    # ========================================================
    # LOG RETRIEVED RESULTS
    # ========================================================

    logger.info(
        "Retrieved %s relevant chunks for: %s",
        len(results),
        rewritten_question
    )

    for rank, result in enumerate(
        results,
        start=1
    ):
        logger.info(
            "ASK RESULT %s | FILE=%s | PAGE=%s | "
            "CHUNK=%s | FINAL=%.4f | "
            "RELEVANCE=%.4f | SIGNATURE=%.4f | "
            "FINANCIAL=%.4f",
            rank,
            result.get(
                "filename"
            ),
            result.get(
                "page_number"
            ),
            result.get(
                "chunk_index"
            ),
            result.get(
                "final_score",
                0.0
            ),
            result.get(
                "relevance_score",
                0.0
            ),
            result.get(
                "signature_score",
                0.0
            ),
            result.get(
                "financial_score",
                0.0
            ),
        )

    # ========================================================
    # BUILD GEMINI CONTEXT
    # ========================================================
    context_parts = []

    for result in results:
        context_parts.append(
            f"""
NAVIGATION METADATA (NOT FACTUAL EVIDENCE):
DOCUMENT LABEL: {result.get("filename", "")}
PAGE: {result.get("page_number", "Not specified")}
CHUNK: {result.get("chunk_index", "Not specified")}
RETRIEVAL RELEVANCE: {result.get("relevance_score", 0.0):.3f}
SEMANTIC SCORE: {result.get("semantic_score", 0.0):.3f}
LEXICAL SCORE: {result.get("lexical_score", 0.0):.3f}
SIGNATURE RELEVANCE: {result.get("signature_score", 0.0):.3f}
FINANCIAL RELEVANCE: {result.get("financial_score", 0.0):.3f}

ACTUAL DOCUMENT CONTENT (FACTUAL EVIDENCE):
{result.get("text", "")}
"""
        )

    context = (
        "\n\n====================\n\n"
        .join(context_parts)
    )

    # ========================================================
    # GEMINI
    # ========================================================
    answer = ask_gemini(
        context,
        rewritten_question,
        current_date=date.today().isoformat(),
    )

    # ========================================================
    # STORE CONVERSATION MESSAGE
    #
    # For now the question is stored unchanged.
    # Actual follow-up rewriting comes in the next step.
    # ========================================================
    conversation_service.add_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question=question,
        answer=answer,
        rewritten_question=rewritten_question,
        sources=results,
    )

    logger.info(
        "Conversation message stored | "
        "conversation=%s | question=%s",
        conversation_id,
        question,
    )

    # ========================================================
    # SOURCES
    # ========================================================
    sources = []

    for result in results:
        sources.append(
            {
                "document_id":
                    result.get(
                        "document_id"
                    ),

                "filename":
                    result.get(
                        "filename"
                    ),

                "page_number":
                    result.get(
                        "page_number"
                    ),

                "chunk_index":
                    result.get(
                        "chunk_index"
                    ),

                "text":
                    result.get(
                        "text",
                        ""
                    ),

                "score":
                    result.get(
                        "final_score",
                        0.0
                    ),

                "relevance_score":
                    result.get(
                        "relevance_score",
                        0.0
                    ),

                "semantic_score":
                    result.get(
                        "semantic_score",
                        0.0
                    ),

                "lexical_score":
                    result.get(
                        "lexical_score",
                        0.0
                    ),

                "signature_score":
                    result.get(
                        "signature_score",
                        0.0
                    ),

                "financial_score":
                    result.get(
                        "financial_score",
                        0.0
                    ),
            }
        )

    # ========================================================
    # RESPONSE
    # ========================================================
    return {
        "question":
            question,

        "answer":
            answer,

        "sources":
            sources,

        "conversation_id":
            conversation_id,
    }


# ============================================================
# DOCUMENT LIST
# ============================================================

def _backfill_agreement_analysis(documents):
    documents_by_id = {
        document.get("id"): document
        for document in documents
        if document.get("id")
    }

    if not documents_by_id:
        return

    document_text = {
        document_id: []
        for document_id in documents_by_id
    }

    for metadata in vector_store.metadata.values():
        document_id = metadata.get("document_id")

        if document_id in document_text:
            text = metadata.get("text")
            if text:
                document_text[document_id].append(
                    str(text)
                )

    metadata_changed = False

    for document_id, text_parts in document_text.items():
        document = documents_by_id[document_id]
        analysis = analyze_agreement(
            " ".join(text_parts),
            expiry_override=document.get("expiry_date"),
        )
        previous_analysis = document.get("analysis")

        if analysis != previous_analysis:
            document["analysis"] = analysis
            metadata_changed = True

        if analysis.get("expiry_date"):
            if document.get("expiry_date") != analysis["expiry_date"]:
                document["expiry_date"] = analysis["expiry_date"]
                metadata_changed = True

            if not document.get("agreement_type"):
                document["agreement_type"] = "Agreement"
                metadata_changed = True

    if metadata_changed:
        document_manager.save_documents(
            documents
        )

@app.get("/documents")
def list_documents(
    current_user: dict = Depends(
        get_current_user
    )
):

    user_id = current_user["id"]

    documents = (
        document_manager
        .get_documents()
    )

    _backfill_agreement_analysis(
        documents
    )

    user_documents = [
        document
        for document in documents
        if document.get(
            "user_id"
        ) == user_id
    ]

    return {

        "total_documents":
            len(user_documents),

        "documents":
            user_documents,
    }


# ============================================================
# CONSISTENCY
# ============================================================

@app.get("/documents/consistency")
def document_consistency(
    current_user: dict = Depends(
        get_current_user
    )
):

    return vector_store.check_consistency(
        user_id=current_user["id"]
    )


# ============================================================
# DELETE DOCUMENT
# ============================================================

@app.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    current_user: dict = Depends(
        get_current_user
    ),
):

    user_id = current_user["id"]

    logger.info(
        "Delete request received: document=%s | user=%s",
        document_id,
        user_id
    )

    # ============================================================
    # FIND USER DOCUMENT
    # ============================================================

    documents = (
        document_manager
        .get_documents()
    )

    document = next(
        (
            item
            for item in documents
            if item.get("id")
            == document_id
            and item.get("user_id")
            == user_id
        ),
        None
    )

    if document is None:

        logger.warning(
            "Document not found: document=%s | user=%s",
            document_id,
            user_id
        )

        return {

            "success":
                False,

            "message":
                "Document not found.",
        }

    filename = document.get(
        "filename",
        ""
    )

    stored_filename = document.get(
        "stored_filename",
        filename
    )

    # ============================================================
    # DELETE FAISS
    # ============================================================

    try:

        faiss_deleted = (
            vector_store
            .delete_document(
                document_id
            )
        )

    except Exception as exc:

        logger.exception(
            "FAISS deletion failed: %s",
            document_id
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Document deletion stopped because "
                "FAISS cleanup failed."
            )
        ) from exc

    # ============================================================
    # DELETE DOCUMENT METADATA
    # ============================================================

    deleted_document = (
        document_manager
        .delete_document(
            document_id
        )
    )

    if deleted_document is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "FAISS vectors were deleted, but document "
                "metadata could not be removed."
            )
        )

    # ============================================================
    # DELETE PHYSICAL PDF
    # ============================================================

    file_path = (
        UPLOAD_FOLDER
        / stored_filename
    )

    physical_file_deleted = False

    if file_path.exists():

        try:

            file_path.unlink()

            physical_file_deleted = True

        except Exception as exc:

            logger.exception(
                "Could not delete physical PDF: %s",
                file_path
            )

            return {

                "success":
                    True,

                "message":
                    (
                        "Document deleted from RAG successfully, "
                        "but physical PDF cleanup failed."
                    ),

                "document":
                    deleted_document,

                "faiss_vectors_deleted":
                    faiss_deleted,

                "physical_file_deleted":
                    False,

                "physical_file_error":
                    str(exc),
            }

    else:

        logger.warning(
            "Physical PDF not found: %s",
            file_path
        )

    return {

        "success":
            True,

        "message":
            "Document deleted successfully.",

        "document":
            deleted_document,

        "faiss_vectors_deleted":
            faiss_deleted,

        "physical_file_deleted":
            physical_file_deleted,
    }


# ============================================================
# CONVERSATION HISTORY
# ============================================================

@app.get("/conversations")
def list_conversations(
    current_user: dict = Depends(
        get_current_user
    ),
):

    conversations = (
        conversation_service
        .list_conversations(
            current_user["id"]
        )
    )

    return {
        "total_conversations": len(conversations),
        "conversations": conversations,
    }


@app.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: str,
    current_user: dict = Depends(
        get_current_user
    ),
):

    messages = (
        conversation_service
        .get_conversation_messages(
            current_user["id"],
            conversation_id,
        )
    )

    if messages is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    return {
        "conversation_id": conversation_id,
        "messages": messages,
    }


@app.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    current_user: dict = Depends(
        get_current_user
    ),
):

    deleted = (
        conversation_service
        .clear_conversation(
            current_user["id"],
            conversation_id,
        )
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    return {
        "success": True,
        "conversation_id": conversation_id,
    }


# ============================================================
# SSO
# ============================================================

@app.get("/auth/sso")
async def sso_login():

    raise HTTPException(
        status_code=501,
        detail=(
            "SSO login not implemented yet. "
            "Please use the demo login."
        ),
    )


# ============================================================
# CUSTOM OPENAPI
# ============================================================

def custom_openapi():

    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version="1.0.0",
        description="PDF RAG Backend API",
        routes=app.routes,
    )

    components = (
        schema
        .get("components", {})
        .get("schemas", {})
    )

    for component in components.values():

        properties = component.get(
            "properties",
            {}
        )

        for prop in properties.values():

            if (
                prop.get("type")
                == "array"
                and isinstance(
                    prop.get("items"),
                    dict
                )
            ):

                items = prop["items"]

                if (
                    items.get(
                        "contentMediaType"
                    )
                    == "application/octet-stream"
                ):

                    items.pop(
                        "contentMediaType",
                        None
                    )

                    items["type"] = "string"
                    items["format"] = "binary"

            elif (
                prop.get("type")
                == "string"
                and prop.get(
                    "contentMediaType"
                )
                == "application/octet-stream"
            ):

                prop.pop(
                    "contentMediaType",
                    None
                )

                prop["format"] = "binary"

    app.openapi_schema = schema

    return app.openapi_schema


app.openapi = custom_openapi
