import json
import logging
import threading
import re
import math
from pathlib import Path
from collections import Counter

import faiss
import numpy as np

from services.embedding_service import embedding_service


logger = logging.getLogger("uvicorn.error")

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "indexes"

FAISS_INDEX_FILE = INDEX_DIR / "faiss.index"
METADATA_FILE = INDEX_DIR / "faiss_metadata.json"

# ============================================================
# SEARCH CONFIGURATION
# ============================================================
DEFAULT_TOP_K = 5
MIN_CANDIDATE_K = 100
CANDIDATE_MULTIPLIER = 20
RRF_K = 60
MAX_PER_DOCUMENT_GENERAL = 2
MAX_PER_DOCUMENT_FINANCIAL = 3
MAX_PER_DOCUMENT_SIGNATURE = 4


# ============================================================
# SIGNATURE SCORE
# ============================================================

def calculate_signature_score(text):
    """
    Score explicit signature/signatory evidence in document text.

    This score is retrieval guidance only. A high score does not
    by itself prove who signed; the final answer must still be
    supported by the actual document content.
    """
    if not text:
        return 0.0

    normalized = re.sub(
        r"\s+",
        " ",
        str(text)
    ).strip().lower()

    esign_matches = list(
        re.finditer(
            r"\be\s*[- ]?sign\b|\belectronic(?:ally)?\s+sign(?:ed|ature)?\b|\bdigitally\s+sign(?:ed|ature)?\b",
            normalized
        )
    )

    signdesk_matches = list(
        re.finditer(
            r"\bsigndesk\b|\bsign\s*desk\b",
            normalized
        )
    )

    executed_matches = list(
        re.finditer(
            r"\bexecuted\s+by\b|\bsigned\s+by\b",
            normalized
        )
    )

    behalf_matches = list(
        re.finditer(
            r"\bfor\s+and\s+on\s+behalf\s+of\b",
            normalized
        )
    )

    signature_matches = list(
        re.finditer(
            r"\bsignature\s*:",
            normalized
        )
    )

    name_matches = list(
        re.finditer(
            r"\bname\s*:",
            normalized
        )
    )

    title_matches = list(
        re.finditer(
            r"\btitle\s*:",
            normalized
        )
    )

    def nearby(matches, position, distance=500):
        return any(
            abs(match.start() - position) <= distance
            for match in matches
        )

    # --------------------------------------------------------
    # Electronic signature evidence
    # --------------------------------------------------------
    if esign_matches:
        explicit_name = any(
            nearby(name_matches, match.start())
            for match in esign_matches
        )

        if signdesk_matches and explicit_name:
            return 0.98

        if explicit_name:
            return 0.95

        if signdesk_matches:
            return 0.90

        return 0.80

    # --------------------------------------------------------
    # Explicit execution evidence
    # --------------------------------------------------------
    if executed_matches:
        explicit_name = any(
            nearby(name_matches, match.start())
            for match in executed_matches
        )

        if explicit_name:
            return 0.95

        return 0.75

    # --------------------------------------------------------
    # Actual signature block
    # --------------------------------------------------------
    for behalf in behalf_matches:
        behalf_pos = behalf.start()

        nearby_signature = nearby(
            signature_matches,
            behalf_pos
        )

        nearby_name = nearby(
            name_matches,
            behalf_pos
        )

        nearby_title = nearby(
            title_matches,
            behalf_pos
        )

        if nearby_signature and nearby_name:
            return 1.0

        if nearby_name and nearby_title:
            return 0.90

        if nearby_signature:
            return 0.75

        if nearby_name:
            return 0.70

    # --------------------------------------------------------
    # Signed / sealed / delivered
    # --------------------------------------------------------
    if re.search(
        r"\bsigned\s+sealed\s+and\s+delivered\b",
        normalized
    ):
        if name_matches and signature_matches:
            return 0.95

        if name_matches:
            return 0.85

    # --------------------------------------------------------
    # Authorized signatory
    # --------------------------------------------------------
    if re.search(
        r"\bauthori[sz]ed\s+signatory\b",
        normalized
    ):
        if name_matches or signature_matches:
            return 0.85

        return 0.60

    # --------------------------------------------------------
    # Signed by
    # --------------------------------------------------------
    if re.search(
        r"\bsigned\s+by\b",
        normalized
    ):
        if name_matches:
            return 0.80

        return 0.50

    # --------------------------------------------------------
    # Generic signature fields
    # --------------------------------------------------------
    if signature_matches and name_matches:
        return 0.70

    if signature_matches:
        return 0.30

    if name_matches and (
        esign_matches
        or signdesk_matches
        or executed_matches
    ):
        return 0.85

    return 0.0


class FAISSVectorStore:

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self):

        INDEX_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        self.lock = threading.Lock()

        self.dimension = (
            embedding_service.model
            .get_embedding_dimension()
        )

        self.index = None
        self.metadata = {}
        self.next_vector_id = 1

        self._load()

    # ========================================================
    # LOAD
    # ========================================================

    def _load(self):

        if FAISS_INDEX_FILE.exists():

            logger.info(
                "Loading FAISS index: %s",
                FAISS_INDEX_FILE
            )

            try:

                self.index = faiss.read_index(
                    str(FAISS_INDEX_FILE)
                )

            except Exception as exc:

                logger.exception(
                    "Failed to load FAISS index: %s",
                    exc
                )

                raise RuntimeError(
                    "FAISS index could not be loaded."
                ) from exc

        else:

            logger.info(
                "Creating new FAISS index. "
                "Dimension=%s",
                self.dimension
            )

            base_index = faiss.IndexFlatIP(
                self.dimension
            )

            self.index = faiss.IndexIDMap2(
                base_index
            )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        if METADATA_FILE.exists():

            try:

                with open(
                    METADATA_FILE,
                    "r",
                    encoding="utf-8"
                ) as file:

                    data = json.load(file)

                self.metadata = data.get(
                    "metadata",
                    {}
                )

                self.next_vector_id = int(
                    data.get(
                        "next_vector_id",
                        1
                    )
                )

            except Exception as exc:

                logger.exception(
                    "Failed to load FAISS metadata: %s",
                    exc
                )

                self.metadata = {}
                self.next_vector_id = 1

        else:

            self.metadata = {}
            self.next_vector_id = 1

        # ----------------------------------------------------
        # Never reuse an ID
        # ----------------------------------------------------

        metadata_ids = []

        for vector_id in self.metadata.keys():

            try:

                metadata_ids.append(
                    int(vector_id)
                )

            except (
                TypeError,
                ValueError
            ):

                logger.warning(
                    "Invalid vector ID: %s",
                    vector_id
                )

        max_metadata_id = (
            max(metadata_ids)
            if metadata_ids
            else 0
        )

        self.next_vector_id = max(
            self.next_vector_id,
            max_metadata_id + 1,
            1
        )

        logger.info(
            "FAISS vector count: %s",
            self.index.ntotal
        )

        logger.info(
            "FAISS metadata entries: %s",
            len(self.metadata)
        )

        logger.info(
            "Next FAISS vector ID: %s",
            self.next_vector_id
        )

        if self.index.ntotal != len(self.metadata):

            logger.warning(
                "FAISS consistency warning: "
                "vectors=%s metadata=%s",
                self.index.ntotal,
                len(self.metadata)
            )

    # ========================================================
    # SAVE
    # ========================================================

    def _save(self):

        temp_index_file = (
            FAISS_INDEX_FILE.with_suffix(".tmp")
        )

        faiss.write_index(
            self.index,
            str(temp_index_file)
        )

        temp_index_file.replace(
            FAISS_INDEX_FILE
        )

        data = {
            "next_vector_id": self.next_vector_id,
            "metadata": self.metadata,
        }

        temp_metadata_file = (
            METADATA_FILE.with_suffix(".tmp")
        )

        with open(
            temp_metadata_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False
            )

        temp_metadata_file.replace(
            METADATA_FILE
        )

    # ========================================================
    # ADD DOCUMENT
    # ========================================================

    def add_document(
        self,
        document_id,
        filename,
        chunks,
        user_id=None,
        persist=True,
    ):
        """
        Add document chunks to the FAISS vector store.

        Existing behavior is preserved:
            - Accepts chunks as strings or dictionaries
            - Supports page_number
            - Creates embeddings
            - Adds vectors to FAISS
            - Stores chunk metadata
            - Saves FAISS index + metadata
            - Returns number of added chunks

        Improvements:
            - Validates document_id / filename
            - Safely handles invalid chunks
            - Removes empty chunks
            - Validates embedding count
            - Validates embedding dimension
            - Prevents FAISS / metadata partial updates
            - Adds detailed logging
            - Preserves user_id isolation
            - Avoids modifying next_vector_id until vectors
            are successfully added
        """

        # ========================================================
        # BASIC VALIDATION
        # ========================================================

        if not document_id:
            logger.warning(
                "Cannot add document: document_id is missing."
            )
            return 0

        if not filename:
            logger.warning(
                "Cannot add document: filename is missing. "
                "document_id=%s",
                document_id,
            )
            return 0

        if not chunks:
            logger.info(
                "No chunks supplied for document: %s",
                document_id,
            )
            return 0

        # ========================================================
        # CLEAN CHUNKS
        # ========================================================

        cleaned_chunks = []

        for chunk_index, chunk in enumerate(chunks):

            # ----------------------------------------------------
            # Dictionary chunk
            # ----------------------------------------------------

            if isinstance(chunk, dict):

                chunk_text = chunk.get(
                    "text",
                    ""
                )

                page_number = chunk.get(
                    "page_number"
                )

            # ----------------------------------------------------
            # String / other chunk
            # ----------------------------------------------------

            else:

                chunk_text = chunk
                page_number = None

            # ----------------------------------------------------
            # Safely convert text to string
            # ----------------------------------------------------

            if chunk_text is None:
                continue

            chunk_text = str(
                chunk_text
            ).strip()

            # ----------------------------------------------------
            # Ignore empty chunks
            # ----------------------------------------------------

            if not chunk_text:
                logger.debug(
                    "Skipping empty chunk | "
                    "document=%s | input_chunk=%s",
                    document_id,
                    chunk_index,
                )
                continue

            cleaned_chunks.append(
                {
                    "text": chunk_text,
                    "page_number": page_number,
                }
            )

        # ========================================================
        # NOTHING USABLE
        # ========================================================

        if not cleaned_chunks:

            logger.warning(
                "No usable chunks found for document: %s",
                document_id,
            )

            return 0

        # ========================================================
        # EXTRACT TEXT
        # ========================================================

        chunk_texts = [
            chunk["text"]
            for chunk in cleaned_chunks
        ]

        logger.info(
            "Creating embeddings | "
            "document=%s | filename=%s | chunks=%s | user=%s",
            document_id,
            filename,
            len(chunk_texts),
            user_id,
        )

        # ========================================================
        # CREATE EMBEDDINGS
        # ========================================================

        try:

            embeddings = (
                embedding_service
                .embed_documents(
                    chunk_texts
                )
            )

        except Exception as exc:

            logger.exception(
                "Embedding generation failed | "
                "document=%s | filename=%s | chunks=%s",
                document_id,
                filename,
                len(chunk_texts),
            )

            raise RuntimeError(
                "Failed to create document embeddings."
            ) from exc

        # ========================================================
        # VALIDATE EMBEDDING RESULT
        # ========================================================

        if embeddings is None:

            logger.error(
                "Embedding service returned None | "
                "document=%s",
                document_id,
            )

            raise RuntimeError(
                "Embedding service returned no embeddings."
            )

        # ========================================================
        # CONVERT TO NUMPY
        # ========================================================

        try:

            vectors = np.asarray(
                embeddings,
                dtype="float32"
            )

        except Exception as exc:

            logger.exception(
                "Failed to convert embeddings to numpy array | "
                "document=%s",
                document_id,
            )

            raise RuntimeError(
                "Invalid embedding data returned by embedding service."
            ) from exc

        # ========================================================
        # VALIDATE VECTOR SHAPE
        # ========================================================

        if vectors.ndim != 2:

            logger.error(
                "Invalid embedding shape | "
                "document=%s | shape=%s",
                document_id,
                vectors.shape,
            )

            raise RuntimeError(
                "Embedding service returned vectors "
                "with an invalid shape."
            )

        # --------------------------------------------------------
        # Number of vectors must match number of chunks
        # --------------------------------------------------------

        if vectors.shape[0] != len(
            cleaned_chunks
        ):

            logger.error(
                "Embedding count mismatch | "
                "document=%s | chunks=%s | embeddings=%s",
                document_id,
                len(cleaned_chunks),
                vectors.shape[0],
            )

            raise RuntimeError(
                "Number of embeddings does not match "
                "number of document chunks."
            )

        # --------------------------------------------------------
        # Embedding dimension must match FAISS index
        # --------------------------------------------------------

        if vectors.shape[1] != self.dimension:

            logger.error(
                "Embedding dimension mismatch | "
                "document=%s | expected=%s | actual=%s",
                document_id,
                self.dimension,
                vectors.shape[1],
            )

            raise RuntimeError(
                "Embedding dimension does not match "
                "the FAISS index dimension."
            )

        # ========================================================
        # CHECK FOR INVALID NUMERIC VALUES
        # ========================================================

        if not np.isfinite(
            vectors
        ).all():

            logger.error(
                "Invalid embedding values detected | "
                "document=%s",
                document_id,
            )

            raise RuntimeError(
                "Embedding vectors contain invalid "
                "NaN or infinite values."
            )

        # ========================================================
        # OPTIONAL NORMALIZATION SAFETY
        #
        # IndexFlatIP is used as the similarity metric.
        #
        # We do NOT normalize here because your existing
        # embedding_service may already perform normalization.
        #
        # Changing that behavior here could alter existing
        # search results.
        # ========================================================

        # ========================================================
        # ADD TO FAISS + METADATA ATOMICALLY
        # ========================================================

        with self.lock:

            # ----------------------------------------------------
            # Reserve IDs without changing next_vector_id yet
            # ----------------------------------------------------

            start_vector_id = (
                self.next_vector_id
            )

            vector_ids = list(
                range(
                    start_vector_id,
                    start_vector_id
                    + len(cleaned_chunks)
                )
            )

            ids = np.asarray(
                vector_ids,
                dtype="int64"
            )

            # ----------------------------------------------------
            # Prepare metadata first
            #
            # This allows us to construct the complete metadata
            # before changing the FAISS index.
            # ----------------------------------------------------

            metadata_entries = {}

            for index, chunk in enumerate(
                cleaned_chunks
            ):

                vector_id = vector_ids[index]

                metadata_entries[
                    str(vector_id)
                ] = {

                    "document_id":
                        document_id,

                    "filename":
                        filename,

                    "chunk_index":
                        index,

                    "page_number":
                        chunk.get(
                            "page_number"
                        ),

                    "text":
                        chunk["text"],

                    "user_id":
                        user_id,
                }

            # ----------------------------------------------------
            # Add vectors to FAISS
            # ----------------------------------------------------

            try:

                self.index.add_with_ids(
                    vectors,
                    ids
                )

            except Exception as exc:

                logger.exception(
                    "Failed to add vectors to FAISS | "
                    "document=%s | vectors=%s",
                    document_id,
                    len(vector_ids),
                )

                raise RuntimeError(
                    "Failed to add document vectors to FAISS."
                ) from exc

            # ----------------------------------------------------
            # Update metadata
            # ----------------------------------------------------

            try:

                self.metadata.update(
                    metadata_entries
                )

                self.next_vector_id = (
                    start_vector_id
                    + len(vector_ids)
                )

            except Exception as exc:

                logger.exception(
                    "Failed to update FAISS metadata | "
                    "document=%s",
                    document_id,
                )

                # ------------------------------------------------
                # Roll back vectors that were just inserted.
                # ------------------------------------------------

                try:

                    self.index.remove_ids(
                        ids
                    )

                except Exception:

                    logger.exception(
                        "CRITICAL: Failed to rollback "
                        "FAISS vectors | document=%s",
                        document_id,
                    )

                raise RuntimeError(
                    "Failed to update document metadata."
                ) from exc

            # ----------------------------------------------------
            # Save index + metadata
            # ----------------------------------------------------

            if persist:
                try:
                    self._save()

                except Exception as exc:
                    logger.exception(
                        "Failed to persist FAISS index/metadata | "
                        "document=%s",
                        document_id,
                    )

                    # ------------------------------------------------
                    # Roll back in-memory changes.
                    #
                    # Important:
                    # _save() uses temporary files, so an existing
                    # persisted index should remain intact if the
                    # temporary replacement was not completed.
                    # ------------------------------------------------

                    try:
                        self.index.remove_ids(
                            ids
                        )

                    except Exception:
                        logger.exception(
                            "CRITICAL: Failed to rollback "
                            "FAISS vectors after save failure | "
                            "document=%s",
                            document_id,
                        )

                    for vector_id in vector_ids:
                        self.metadata.pop(
                            str(vector_id),
                            None
                        )

                    self.next_vector_id = (
                        start_vector_id
                    )

                    raise RuntimeError(
                        "Failed to save document "
                        "to the vector store."
                    ) from exc

        # ========================================================
        # SUCCESS LOGGING
        # ========================================================

        logger.info(
            "Added document successfully | "
            "document=%s | filename=%s | "
            "chunks=%s | vectors=%s | "
            "user=%s | next_vector_id=%s",
            document_id,
            filename,
            len(cleaned_chunks),
            len(vector_ids),
            user_id,
            self.next_vector_id,
        )

        # ========================================================
        # RETURN EXISTING CONTRACT
        # ========================================================

        return len(cleaned_chunks)

    # ========================================================
    # ADD DOCUMENTS BATCH
    # ========================================================

    def add_documents_batch(self, documents, user_id=None, persist=True):
        """
        Add multiple prepared documents in one embedding call and one
        FAISS mutation/save.

        documents:
            [
                {
                    "document_id": str,
                    "filename": str,
                    "chunks": list[dict|str],
                    "user_id": optional,
                },
                ...
            ]
        """
        if not documents:
            return []

        prepared = []
        all_texts = []

        for item in documents:
            document_id = item.get("document_id")
            filename = item.get("filename")
            chunks = item.get("chunks") or []
            item_user_id = item.get("user_id", user_id)

            if not document_id or not filename:
                raise ValueError("Batch document is missing document_id or filename.")

            cleaned = []
            for chunk_index, chunk in enumerate(chunks):
                if isinstance(chunk, dict):
                    text = chunk.get("text", "")
                    page_number = chunk.get("page_number")
                else:
                    text = chunk
                    page_number = None

                if text is None:
                    continue
                text = str(text).strip()
                if not text:
                    continue

                cleaned.append({
                    "text": text,
                    "page_number": page_number,
                })

            if not cleaned:
                raise ValueError(
                    f"No usable chunks found for document: {filename}"
                )

            prepared.append({
                "document_id": document_id,
                "filename": filename,
                "chunks": cleaned,
                "user_id": item_user_id,
            })
            all_texts.extend(chunk["text"] for chunk in cleaned)

        logger.info(
            "Creating batch embeddings | documents=%s | chunks=%s",
            len(prepared),
            len(all_texts),
        )

        embeddings = embedding_service.embed_documents(all_texts)
        vectors = np.asarray(embeddings, dtype="float32")

        if vectors.ndim != 2 or vectors.shape[0] != len(all_texts):
            raise RuntimeError(
                "Batch embedding count or shape does not match document chunks."
            )
        if vectors.shape[1] != self.dimension:
            raise RuntimeError(
                "Batch embedding dimension does not match the FAISS index."
            )
        if not np.isfinite(vectors).all():
            raise RuntimeError("Batch embedding vectors contain invalid values.")

        added = []
        offset = 0

        with self.lock:
            start_vector_id = self.next_vector_id
            total_vectors = len(all_texts)
            vector_ids = list(
                range(start_vector_id, start_vector_id + total_vectors)
            )
            ids = np.asarray(vector_ids, dtype="int64")

            metadata_entries = {}
            cursor = 0

            for item in prepared:
                item_count = len(item["chunks"])
                item_ids = vector_ids[cursor:cursor + item_count]

                for local_index, chunk in enumerate(item["chunks"]):
                    vector_id = item_ids[local_index]
                    metadata_entries[str(vector_id)] = {
                        "document_id": item["document_id"],
                        "filename": item["filename"],
                        "chunk_index": local_index,
                        "page_number": chunk.get("page_number"),
                        "text": chunk["text"],
                        "user_id": item["user_id"],
                    }

                added.append({
                    "document_id": item["document_id"],
                    "filename": item["filename"],
                    "chunks_added": item_count,
                })
                cursor += item_count

            try:
                self.index.add_with_ids(vectors, ids)
                self.metadata.update(metadata_entries)
                self.next_vector_id = start_vector_id + total_vectors

                if persist:
                    self._save()

            except Exception as exc:
                logger.exception(
                    "Batch FAISS update failed; rolling back %s vectors.",
                    total_vectors,
                )
                try:
                    self.index.remove_ids(ids)
                except Exception:
                    logger.exception("CRITICAL: failed to rollback batch vectors")

                for vector_id in vector_ids:
                    self.metadata.pop(str(vector_id), None)
                self.next_vector_id = start_vector_id
                raise RuntimeError("Failed to persist batch document vectors.") from exc

        logger.info(
            "Batch documents indexed | documents=%s | vectors=%s",
            len(prepared),
            total_vectors,
        )
        return added

    # ========================================================
    # TOKENIZER
    # ========================================================

    def _tokenize(self, text):

        if not text:
            return []

        text = str(text).lower()

        text = text.replace(
            "&",
            " and "
        )

        text = text.replace(
            "/",
            " "
        )

        text = text.replace(
            "-",
            " "
        )

        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        tokens = re.findall(
            r"\b[a-z0-9]+\b",
            text
        )

        aliases = {

            "rental": "rent",
            "renting": "rent",
            "rented": "rent",

            "prices": "price",
            "pricing": "price",
            "priced": "price",

            "costs": "cost",
            "costing": "cost",

            "charges": "charge",
            "charging": "charge",

            "amounts": "amount",

            "rates": "rate",

            "warehouses": "warehouse",
            "warehouse": "warehouse",

            "offices": "office",
            "office": "office",

            "spaces": "space",
            "space": "space",
        }

        return [
            aliases.get(
                token,
                token
            )
            for token in tokens
        ]

    # ========================================================
    # LEXICAL SEARCH
    # ========================================================

    def _lexical_search(
        self,
        query,
        top_k=20,
        user_id=None,
        filename=None,
    ):

        query_tokens = self._tokenize(
            query
        )

        if not query_tokens:
            return []

        documents = []

        for vector_id, metadata in (
            self.metadata.items()
        ):

            if user_id is not None:

                if metadata.get(
                    "user_id"
                ) != user_id:
                    continue

            if filename is not None:
                if metadata.get(
                    "filename"
                ) != filename:
                    continue

            tokens = self._tokenize(
                metadata.get(
                    "text",
                    ""
                )
            )

            if tokens:

                documents.append(
                    (
                        int(vector_id),
                        tokens
                    )
                )

        if not documents:
            return []

        total_documents = len(
            documents
        )

        document_frequency = Counter()

        for _, tokens in documents:

            for token in set(tokens):

                document_frequency[token] += 1

        average_length = (
            sum(
                len(tokens)
                for _, tokens in documents
            )
            / total_documents
        )

        k1 = 1.5
        b = 0.75

        query_term_counts = Counter(
            query_tokens
        )

        scored_documents = []

        for vector_id, tokens in documents:

            token_counts = Counter(
                tokens
            )

            document_length = len(
                tokens
            )

            score = 0.0

            for term, query_count in (
                query_term_counts.items()
            ):

                term_frequency = (
                    token_counts.get(
                        term,
                        0
                    )
                )

                if term_frequency == 0:
                    continue

                df = document_frequency.get(
                    term,
                    0
                )

                idf = math.log(
                    1
                    + (
                        total_documents
                        - df
                        + 0.5
                    )
                    / (
                        df
                        + 0.5
                    )
                )

                denominator = (
                    term_frequency
                    + k1
                    * (
                        1
                        - b
                        + b
                        * (
                            document_length
                            / average_length
                        )
                    )
                )

                term_score = (
                    idf
                    * (
                        term_frequency
                        * (k1 + 1)
                        / denominator
                    )
                )

                score += (
                    term_score
                    * query_count
                )

            if score <= 0:
                continue

            metadata = self.metadata.get(
                str(vector_id)
            )

            if not metadata:
                continue

            scored_documents.append(
                {
                    "score": float(score),
                    "vector_id": vector_id,
                    "document_id":
                        metadata["document_id"],
                    "filename":
                        metadata["filename"],
                    "chunk_index":
                        metadata["chunk_index"],
                    "page_number":
                        metadata.get(
                            "page_number"
                        ),
                    "text":
                        metadata["text"],
                }
            )

        scored_documents.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return scored_documents[:top_k]

    # ========================================================
    # SEMANTIC SEARCH
    # ========================================================

    def _semantic_search(
        self,
        query,
        top_k=20,
        user_id=None,
    ):
        if not query or not query.strip():
            return []

        if self.index.ntotal <= 0:
            return []

        query_embedding = (
            embedding_service.embed_text(
                query
            )
        )

        if query_embedding is None:
            return []

        query_vector = np.asarray(
            [query_embedding],
            dtype="float32"
        )

        if query_vector.ndim != 2:
            raise ValueError(
                "Query embedding must be a 2D vector."
            )

        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                "Embedding dimension mismatch. "
                f"Expected={self.dimension}, "
                f"Received={query_vector.shape[1]}"
            )

        search_k = min(
            self.index.ntotal,
            max(
                int(top_k),
                1
            )
        )

        scores, vector_ids = (
            self.index.search(
                query_vector,
                search_k
            )
        )

        results = []

        for score, vector_id in zip(
            scores[0],
            vector_ids[0]
        ):
            if vector_id == -1:
                continue

            vector_id = int(
                vector_id
            )

            metadata = self.metadata.get(
                str(vector_id)
            )

            if not metadata:
                continue

            if user_id is not None:
                if metadata.get(
                    "user_id"
                ) != user_id:
                    continue

            results.append(
                {
                    "score": float(score),
                    "vector_id": vector_id,
                    "document_id":
                        metadata.get(
                            "document_id"
                        ),
                    "filename":
                        metadata.get(
                            "filename",
                            ""
                        ),
                    "chunk_index":
                        metadata.get(
                            "chunk_index",
                            0
                        ),
                    "page_number":
                        metadata.get(
                            "page_number"
                        ),
                    "text":
                        metadata.get(
                            "text",
                            ""
                        ),
                }
            )

            if len(results) >= top_k:
                break

        return results

    # ========================================================
    # NORMALIZE TEXT
    # ========================================================

    def _normalize_text(self, text):
        if not text:
            return ""

        text = str(text).lower()

        text = text.replace(
            "&",
            " and "
        )

        text = text.replace(
            "/",
            " "
        )

        text = text.replace(
            "-",
            " "
        )

        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        replacements = {
            "rental": "rent",
            "renting": "rent",
            "rented": "rent",
            "prices": "price",
            "pricing": "price",
            "priced": "price",
            "costs": "cost",
            "costing": "cost",
            "charges": "charge",
            "charging": "charge",
            "amounts": "amount",
            "rates": "rate",
            "warehouses": "warehouse",
            "offices": "office",
            "spaces": "space",
        }

        words = text.split()

        words = [
            replacements.get(
                word,
                word
            )
            for word in words
        ]

        return " ".join(words)

    # ========================================================
    # QUERY RELEVANCE
    # ========================================================

    def _query_relevance_score(
        self,
        query,
        text
    ):

        if not query or not text:
            return 0.0

        normalized_query = (
            self._normalize_text(
                query
            )
        )

        normalized_text = (
            self._normalize_text(
                text
            )
        )

        query_tokens = self._tokenize(
            normalized_query
        )

        text_tokens = self._tokenize(
            normalized_text
        )

        if not query_tokens or not text_tokens:
            return 0.0

        stopwords = {

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

            "can",
            "could",
            "would",
            "should",

            "please",
            "tell",
            "me",
            "give",

            "according",
            "does",
            "do",
            "about",
        }

        query_terms = [
            token
            for token in query_tokens
            if token not in stopwords
            and (
                len(token) > 2
                # Keep short identifiers such as L1, C2, B1.
                or any(
                    character.isdigit()
                    for character in token
                )
            )
        ]

        if not query_terms:
            return 0.0

        text_token_set = set(
            text_tokens
        )

        matched_terms = sum(
            1
            for term in query_terms
            if term in text_token_set
        )

        token_coverage = (
            matched_terms
            / len(query_terms)
        )

        # ----------------------------------------------------
        # Business concepts
        # ----------------------------------------------------

        concept_groups = [

            {
                "terms": {
                    "cost",
                    "price",
                    "rate",
                    "amount",
                    "charge",
                    "rent",
                }
            },

            {
                "terms": {
                    "warehouse",
                }
            },

            {
                "terms": {
                    "office",
                }
            },

            {
                "terms": {
                    "open",
                    "space",
                }
            },
        ]

        query_concepts = []

        for concept in concept_groups:

            if any(
                term in query_terms
                for term in concept["terms"]
            ):
                query_concepts.append(
                    concept
                )

        concept_matches = 0

        for concept in query_concepts:

            if any(
                term in text_token_set
                for term in concept["terms"]
            ):

                concept_matches += 1

        concept_coverage = (
            concept_matches
            / len(query_concepts)
            if query_concepts
            else 0.0
        )

        # ----------------------------------------------------
        # Exact phrase
        # ----------------------------------------------------

        exact_phrase_bonus = 0.0

        if (
            normalized_query
            in normalized_text
        ):
            exact_phrase_bonus = 0.30

        # ----------------------------------------------------
        # Important phrases
        # ----------------------------------------------------

        phrase_bonus = 0.0

        important_phrases = [

            "warehouse rent",
            "rent of warehouse",
            "warehouse rental",

            "office rent",
            "rent of office",

            "cost of rent",
            "rent amount",

            "open space rent",
        ]

        for phrase in important_phrases:

            normalized_phrase = (
                self._normalize_text(
                    phrase
                )
            )

            if (
                normalized_phrase
                in normalized_query
                and normalized_phrase
                in normalized_text
            ):

                phrase_bonus = max(
                    phrase_bonus,
                    0.20
                )

        relevance = (
            token_coverage * 0.40
            + concept_coverage * 0.50
            + phrase_bonus
            + exact_phrase_bonus
        )

        return min(
            1.0,
            relevance
        )

    # ========================================================
    # DOCUMENT HINT
    # ========================================================

    def _document_hint_score(
        self,
        query,
        filename
    ):

        query_text = self._normalize_text(
            query
        )

        file_text = self._normalize_text(
            Path(filename).stem
        )

        if not query_text or not file_text:
            return 0.0

        aliases = {

            "wh": "warehouse",
            "whs": "warehouse",
            "warehouses": "warehouse",

            "off": "office",
            "offs": "office",
            "offices": "office",
        }

        query_tokens = {
            aliases.get(
                token,
                token
            )
            for token in query_text.split()
        }

        file_tokens = {
            aliases.get(
                token,
                token
            )
            for token in file_text.split()
        }

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

            "can",
            "could",
            "would",
            "should",

            "please",
            "tell",
            "me",

            "agreement",
            "document",
            "file",
            "pdf",
        }

        query_tokens -= generic_terms
        file_tokens -= generic_terms

        if not query_tokens or not file_tokens:
            return 0.0

        overlap = (
            query_tokens
            .intersection(
                file_tokens
            )
        )

        if not overlap:
            return 0.0

        query_coverage = (
            len(overlap)
            / len(query_tokens)
        )

        distinctive_tokens = {
            token
            for token in file_tokens
            if token not in {
                "lp",
                "wh",
                "warehouse",
                "office",
                "space",
            }
            and not token.isdigit()
        }

        distinctive_overlap = (
            overlap
            .intersection(
                distinctive_tokens
            )
        )

        score = (
            query_coverage * 0.70
        )

        if distinctive_overlap:
            score += 0.30

        return min(
            1.0,
            score
        )

    # ========================================================
    # TARGET DOCUMENT DETECTION
    # ========================================================

    def _detect_target_documents(
        self,
        query,
        user_id=None
    ):

        query_text = self._normalize_text(
            query
        )

        if not query_text:
            return set()

        aliases = {

            "wh": "warehouse",
            "whs": "warehouse",
            "warehouses": "warehouse",

            "off": "office",
            "offs": "office",
            "offices": "office",
        }

        generic_terms = {

            "agreement",
            "agreements",
            "document",
            "documents",
            "file",
            "files",
            "pdf",
            "contract",
            "contracts",
        }

        query_tokens = set(
            query_text.split()
        )

        query_tokens = {
            aliases.get(
                token,
                token
            )
            for token in query_tokens
        }

        query_tokens = {
            token
            for token in query_tokens
            if token not in generic_terms
            and not token.isdigit()
            and len(token) > 1
        }

        if not query_tokens:
            return set()

        documents = {}

        for metadata in self.metadata.values():

            if user_id is not None:

                if metadata.get(
                    "user_id"
                ) != user_id:
                    continue

            filename = metadata.get(
                "filename",
                ""
            )

            if not filename:
                continue

            filename_tokens = (
                self._normalize_text(
                    Path(filename).stem
                ).split()
            )

            filename_tokens = {
                aliases.get(
                    token,
                    token
                )
                for token in filename_tokens
            }

            filename_tokens = {
                token
                for token in filename_tokens
                if token not in generic_terms
                and not token.isdigit()
                and len(token) > 1
            }

            documents[filename] = (
                filename_tokens
            )

        document_scores = {}

        for filename, filename_tokens in (
            documents.items()
        ):

            if not filename_tokens:
                continue

            overlap = (
                query_tokens
                .intersection(
                    filename_tokens
                )
            )

            if not overlap:
                continue

            overlap_count = len(
                overlap
            )

            overlap_ratio = (
                overlap_count
                / len(filename_tokens)
            )

            query_coverage = (
                overlap_count
                / len(query_tokens)
            )

            distinctive_tokens = {
                token
                for token in filename_tokens
                if token not in {
                    "lp",
                    "wh",
                    "warehouse",
                    "office",
                    "space",
                }
            }

            distinctive_overlap = (
                overlap
                .intersection(
                    distinctive_tokens
                )
            )

            is_strong_match = (
                overlap_count >= 2
                or overlap_ratio >= 0.50
                or query_coverage >= 0.50
                or len(distinctive_overlap) >= 1
            )

            if not is_strong_match:
                continue

            score = (
                (
                    10.0
                    if distinctive_overlap
                    else 0.0
                )
                + overlap_count * 3.0
                + query_coverage * 2.0
                + overlap_ratio
            )

            document_scores[filename] = {
                "score": score,
                "distinctive_overlap":
                    distinctive_overlap,
            }

        if not document_scores:
            return set()

        best_score = max(
            item["score"]
            for item in document_scores.values()
        )

        target_documents = {
            filename
            for filename, item
            in document_scores.items()
            if abs(
                item["score"]
                - best_score
            ) < 0.000001
        }

        logger.info(
            "Target documents for query='%s': %s",
            query,
            list(target_documents)
        )

        return target_documents

    # ========================================================
    # INTENT
    # ========================================================

    def _strip_document_reference(
        self,
        query,
        filenames=None,
    ):
        """
        Remove document-name mentions from a query.

        A filename such as "Bristlecone RPT Agreement_Signed.pdf"
        contributes words like "signed" that would otherwise be read
        as question intent rather than as a document reference.
        """
        if not query:
            return query

        cleaned = str(query)

        for filename in filenames or []:
            if not filename:
                continue

            stem = Path(str(filename)).stem

            variants = {
                stem,
                stem.replace("_", " "),
                stem.replace("-", " "),
                re.sub(r"[_\-]+", " ", stem),
            }

            for variant in sorted(
                variants,
                key=len,
                reverse=True,
            ):
                variant = variant.strip()

                if len(variant) < 4:
                    continue

                cleaned = re.sub(
                    re.escape(variant),
                    " ",
                    cleaned,
                    flags=re.IGNORECASE,
                )

        cleaned = re.sub(
            r"\s+",
            " ",
            cleaned,
        ).strip()

        # Fall back to the original when the query was only a document name.
        return cleaned or str(query)

    def _detect_intent(self, query):
        normalized = self._normalize_text(
            query
        )

        if not normalized:
            return "general"

        signature_terms = [
            "signatory",
            "signatories",
            "signature",
            "signatures",
            "signed",
            "signed by",
            "who signed",
            "who has signed",
            "authorized signatory",
            "authorised signatory",
            "authorized representative",
            "authorised representative",
            "esign",
            "e sign",
            "e-sign",
            "electronic signature",
            "electronically signed",
            "digital signature",
            "digitally signed",
            "signdesk",
            "execution",
            "executed by",
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
            "duration",
            "validity",
            "valid until",
            "agreement period",
            "renewal",
            "renew",
            "active",
            "expired",
        ]

        actualization_terms = [
            "actualization",
            "actualisation",
            "actualize",
            "actualise",
            "actualized",
            "actualised",
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
            "business justification",
            "commercial intent",
            "strategic objective",
            "objective",
            "reason for",
            "why was",
            "why did",
        ]

        transaction_terms = [
            "transaction",
            "nature of transaction",
            "nature of activity",
            "activities",
            "activity",
            "purchase",
            "sale",
            "service",
            "services",
            "design service",
            "manufacturing",
            "contract manufacturing",
            "royalty",
            "procurement",
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

    # ========================================================
    # INTENT SCORE
    # ========================================================
    def _intent_score(
        self,
        query,
        text
    ):
        """
        Calculate how strongly a chunk matches the detected
        query intent.
        IMPORTANT:
        This function ALWAYS returns a float.
        Previously, the function returned None for "general"
        intent, which caused:
            intent_score * 0.05
        to raise:
            TypeError:
            unsupported operand type(s) for *:
            'NoneType' and 'float'
        """

        # ----------------------------------------------------
        # Detect query intent
        # ----------------------------------------------------
        intent = self._detect_intent(
            query
        )

        # ----------------------------------------------------
        # Safety checks
        # ----------------------------------------------------
        if not query or not text:
            return 0.0

        normalized_text = (
            self._normalize_text(
                text
            )
        )

        if not normalized_text:
            return 0.0

        # ----------------------------------------------------
        # SIGNATURE INTENT
        # ----------------------------------------------------
        if intent == "signature":
            terms = [
                "signature",
                "signatory",
                "signed",
                "signed by",
                "authorized signatory",
                "authorised signatory",
                "esign",
                "e sign",
                "witness whereof",
            ]

            count = sum(
                1
                for term in terms
                if term in normalized_text
            )

            return float(
                min(
                    1.0,
                    count * 0.20
                )
            )

        # ----------------------------------------------------
        # COMMERCIAL INTENT
        # ----------------------------------------------------
        if intent == "commercial":
            commercial_terms = {
                "rent",
                "rental",
                "price",
                "pricing",
                "cost",
                "rate",
                "amount",
                "charge",
                "monthly",
                "annual",
                "annum",
                "markup",
                "margin",
                "payment",
                "payable",
                "fee",
                "consideration",
            }

            text_tokens = set(
                self._tokenize(
                    normalized_text
                )
            )

            matched_terms = (
                text_tokens.intersection(
                    commercial_terms
                )
            )

            if not matched_terms:
                return 0.0

            return float(
                min(
                    1.0,
                    len(
                        matched_terms
                    ) * 0.15
                )
            )

        # ----------------------------------------------------
        # LIFECYCLE INTENT
        # ----------------------------------------------------
        if intent == "lifecycle":
            terms = {
                "effective",
                "commencement",
                "start",
                "expiry",
                "expire",
                "termination",
                "term",
                "tenure",
                "duration",
                "validity",
                "renewal",
                "renew",
                "period",
                "until",
            }
            matched = set(
                self._tokenize(normalized_text)
            ).intersection(terms)
            return float(
                min(1.0, len(matched) * 0.15)
            )

        # ----------------------------------------------------
        # PURPOSE INTENT
        # ----------------------------------------------------
        if intent == "purpose":
            terms = {
                "purpose",
                "objective",
                "business",
                "rationale",
                "need",
                "intent",
                "strategic",
                "commercial",
                "reason",
                "scope",
            }
            matched = set(
                self._tokenize(normalized_text)
            ).intersection(terms)
            return float(
                min(1.0, len(matched) * 0.15)
            )

        # ----------------------------------------------------
        # TRANSACTION INTENT
        # ----------------------------------------------------
        if intent == "transaction":
            terms = {
                "purchase",
                "sale",
                "service",
                "services",
                "manufacturing",
                "royalty",
                "procurement",
                "design",
                "supply",
                "transportation",
            }
            matched = set(
                self._tokenize(normalized_text)
            ).intersection(terms)
            return float(
                min(1.0, len(matched) * 0.15)
            )

        # ----------------------------------------------------
        # ACTUALIZATION INTENT
        # ----------------------------------------------------
        if intent == "actualization":
            terms = {
                "actualization",
                "actualisation",
                "actualize",
                "actualise",
                "actualized",
                "actualised",
            }
            matched = set(
                self._tokenize(normalized_text)
            ).intersection(terms)
            return float(
                min(1.0, len(matched) * 0.25)
            )

        # ----------------------------------------------------
        # GENERAL INTENT
        #
        # CRITICAL FIX:
        #
        # Previously this function reached the end here and
        # implicitly returned None.
        #
        # Now it explicitly returns 0.0.
        # ----------------------------------------------------
        return 0.0

    # ========================================================
    # SIGNATURE SCORE
    # ========================================================

    def _signature_score(
        self,
        query,
        text
    ):

        if (
            self._detect_intent(query)
            != "signature"
        ):
            return 0.0

        return calculate_signature_score(
            text
        )

    # ============================================================
    # FINANCIAL SCORE
    # ============================================================
    def _financial_score(
        self,
        query,
        text
    ):
        """
        Calculate financial relevance for financial questions.
        The score is always a float between 0.0 and 1.0.
        Financial scoring is applied only when the query intent
        is financial.
        Numbers alone are NOT considered financial evidence.
        Generic contract terms such as lease, payment, payable,
        and fee are intentionally excluded because they can occur
        in non-financial sections.
        """
        # --------------------------------------------------------
        # Only apply financial scoring to financial questions
        # --------------------------------------------------------
        if (
            self._detect_intent(query)
            != "commercial"
        ):
            return 0.0
        if not text:
            return 0.0

        # --------------------------------------------------------
        # Normalize query and document text
        # --------------------------------------------------------
        normalized_text = self._normalize_text(
            text
        )

        normalized_query = self._normalize_text(
            query
        )

        if not normalized_text:
            return 0.0

        if not normalized_query:
            return 0.0

        # --------------------------------------------------------
        # Financial terminology
        #
        # These are actual financial/rental concepts.
        #
        # Deliberately NOT including:
        # lease
        # payment
        # payable
        # fee
        #
        # because these can occur in unrelated contract sections.
        # --------------------------------------------------------
        financial_terms = {
            "rent",
            "rental",
            "price",
            "rate",
            "cost",
            "amount",
            "charge",
            "monthly",
            "annual",
            "annum",
        }

        # --------------------------------------------------------
        # Tokenize
        # --------------------------------------------------------
        text_tokens = set(
            self._tokenize(
                normalized_text
            )
        )

        query_tokens = set(
            self._tokenize(
                normalized_query
            )
        )

        if not text_tokens:
            return 0.0

        # --------------------------------------------------------
        # Financial terms found in document
        # --------------------------------------------------------
        financial_matches = (
            text_tokens.intersection(
                financial_terms
            )
        )

        # --------------------------------------------------------
        # CRITICAL SAFETY CHECK
        #
        # If there are no actual financial terms in the chunk,
        # financial score MUST be zero.
        #
        # Therefore a chunk containing only:
        #
        # Name: ABC
        # Date: 01/04/2024
        # Signature: ...
        #
        # will NOT receive financial points merely because it
        # contains numbers.
        # --------------------------------------------------------
        if not financial_matches:
            return 0.0

        score = 0.0

        # --------------------------------------------------------
        # Financial terminology score
        #
        # Maximum = 0.60
        # --------------------------------------------------------
        score += min(
            0.60,
            len(financial_matches) * 0.15
        )

        # --------------------------------------------------------
        # Query-specific financial terminology
        #
        # Example:
        #
        # Query:
        # "What is the rent of Nagpur Warehouse?"
        #
        # Query financial term:
        # "rent"
        #
        # If the chunk also contains "rent", give additional
        # relevance.
        # --------------------------------------------------------
        query_financial_terms = (
            query_tokens.intersection(
                financial_terms
            )
        )

        matching_query_financial_terms = (
            financial_matches.intersection(
                query_financial_terms
            )
        )

        if matching_query_financial_terms:
            score += min(
                0.20,
                len(
                    matching_query_financial_terms
                ) * 0.10
            )

        # --------------------------------------------------------
        # Strong financial/rental phrases
        # --------------------------------------------------------
        strong_phrases = [
            "warehouse rent",
            "rent of warehouse",
            "warehouse rental",
            "monthly rent",
            "rent per month",
            "rent per annum",
            "annual rent",
            "rental amount",
            "rental rate",
            "rate applicable",
            "rates applicable",
            "cost per month",
            "cost per annum",
            "monthly cost",
            "annual cost",
            "total cost",
            "total costs",
        ]

        for phrase in strong_phrases:
            normalized_phrase = (
                self._normalize_text(
                    phrase
                )
            )

            if (
                normalized_phrase
                in normalized_text
            ):
                score += 0.20
                break

        # --------------------------------------------------------
        # Currency evidence
        #
        # Currency strengthens an already financial chunk.
        # Currency by itself cannot create a financial score.
        # --------------------------------------------------------
        if re.search(
            r"(â‚¹|rs\.?|inr|\$|â‚¬|Â£)",
            text,
            re.IGNORECASE
        ):
            score += 0.15

        # --------------------------------------------------------
        # Amount-like number
        #
        # Small supporting contribution only.
        # --------------------------------------------------------
        if re.search(
            r"\b\d[\d,]*(?:\.\d+)?\b",
            text
        ):
            score += 0.05

        # --------------------------------------------------------
        # Final score
        #
        # Always return a numeric float.
        # --------------------------------------------------------
        return float(
            min(
                1.0,
                score
            )
        )

    # ========================================================
    # HYBRID SEARCH
    # ========================================================

    def search(
        self,
        query,
        top_k=5,
        user_id=None,
        resolved_filename=None,
    ):

        if not query or not query.strip():
            return []

        if self.index.ntotal == 0:
            return []

        intent = self._detect_intent(
            query
        )

        logger.info(
            "RAG query intent: %s | query=%s",
            intent,
            query
        )

        target_documents = set()

        if resolved_filename:
            target_documents = {
                resolved_filename
            }
        else:
            target_documents = (
                self._detect_target_documents(
                    query,
                    user_id=user_id
                )
            )

        has_explicit_document = bool(
            target_documents
        )

        # ----------------------------------------------------
        # Retrieve MORE candidates.
        #
        # This is particularly important for signature
        # and financial questions.
        # ----------------------------------------------------

        candidate_k = min(
            self.index.ntotal,
            max(
                MIN_CANDIDATE_K,
                top_k * CANDIDATE_MULTIPLIER
            )
        )

        semantic_results = (
            self._semantic_search(
                query,
                top_k=candidate_k,
                user_id=user_id
            )
        )

        lexical_results = (
            self._lexical_search(
                query,
                top_k=candidate_k,
                user_id=user_id
            )
        )

        # ----------------------------------------------------
        # RRF
        # ----------------------------------------------------

        fusion_scores = {}
        result_lookup = {}

        rrf_k = RRF_K

        for rank, result in enumerate(
            semantic_results,
            start=1
        ):

            vector_id = result[
                "vector_id"
            ]

            fusion_scores.setdefault(
                vector_id,
                0.0
            )

            fusion_scores[
                vector_id
            ] += (
                1.0
                / (
                    rrf_k + rank
                )
            )

            result_lookup[
                vector_id
            ] = dict(result)

            result_lookup[
                vector_id
            ]["semantic_score"] = float(
                result.get(
                    "score",
                    0.0
                )
            )

        for rank, result in enumerate(
            lexical_results,
            start=1
        ):

            vector_id = result[
                "vector_id"
            ]

            fusion_scores.setdefault(
                vector_id,
                0.0
            )

            fusion_scores[
                vector_id
            ] += (
                1.0
                / (
                    rrf_k + rank
                )
            )

            if vector_id not in result_lookup:

                result_lookup[
                    vector_id
                ] = dict(result)

                result_lookup[
                    vector_id
                ]["semantic_score"] = 0.0

            result_lookup[
                vector_id
            ]["lexical_score"] = float(
                result.get(
                    "score",
                    0.0
                )
            )

        # ----------------------------------------------------
        # Score candidates
        # ----------------------------------------------------

        ranked_results = []

        for vector_id, fusion_score in (
            fusion_scores.items()
        ):

            result = result_lookup.get(
                vector_id
            )

            if not result:
                continue

            filename = result.get(
                "filename",
                ""
            )

            # ------------------------------------------------
            # Explicit document restriction
            # ------------------------------------------------

            if (
                has_explicit_document
                and filename
                not in target_documents
            ):
                continue

            semantic_score = float(
                result.get(
                    "semantic_score",
                    0.0
                )
            )

            lexical_score = float(
                result.get(
                    "lexical_score",
                    0.0
                )
            )

            relevance_score = (
                self._query_relevance_score(
                    query,
                    result["text"]
                )
            )

            document_score = (
                self._document_hint_score(
                    query,
                    filename
                )
            )

            intent_score = (
                self._intent_score(
                    query,
                    result["text"]
                )
            )

            signature_score = (
                self._signature_score(
                    query,
                    result["text"]
                )
            )

            financial_score = (
                self._financial_score(
                    query,
                    result["text"]
                )
            )

            lifecycle_score = self._lifecycle_score(
                query,
                result["text"]
            )

            purpose_score = self._purpose_score(
                query,
                result["text"]
            )

            transaction_score = self._transaction_score(
                query,
                result["text"]
            )

            document_reference_boost = 0.0

            if (
                has_explicit_document
                and filename
                in target_documents
            ):

                document_reference_boost = 0.15

            elif document_score >= 0.50:

                document_reference_boost = 0.10

            elif document_score >= 0.25:

                document_reference_boost = 0.05

            # ------------------------------------------------
            # Final score
            #
            # IMPORTANT:
            # Signature and financial intent are only strong
            # when the question actually asks for them.
            # ------------------------------------------------

            intent_bonus = 0.0
            
            intent = self._detect_intent(
                query
            )

            if intent == "signature":
                intent_bonus = (
                    signature_score * 0.15
                )

            elif intent == "commercial":
                intent_bonus = (
                    financial_score * 0.15
                )

            elif intent == "lifecycle":
                intent_bonus = (
                    lifecycle_score * 0.15
                )

            elif intent == "purpose":
                intent_bonus = (
                    purpose_score * 0.15
                )

            elif intent == "transaction":
                intent_bonus = (
                    transaction_score * 0.15
                )

            elif intent == "actualization":
                intent_bonus = (
                    self._intent_score(
                        query,
                        result["text"]
                    ) * 0.15
                )

            final_score = (
                semantic_score * 0.40
                + lexical_score * 0.10
                + relevance_score * 0.25
                + intent_score * 0.05
                + intent_bonus
                + signature_score * 0.025
                + financial_score * 0.025
                + document_reference_boost
            )

            result["fusion_score"] = float(
                final_score
            )

            result["semantic_score"] = (
                semantic_score
            )

            result["lexical_score"] = (
                lexical_score
            )

            result["relevance_score"] = (
                float(relevance_score)
            )

            result["document_score"] = (
                float(document_score)
            )

            result["document_reference_boost"] = (
                float(
                    document_reference_boost
                )
            )

            result["intent_score"] = (
                float(intent_score)
            )

            result["signature_score"] = (
                float(signature_score)
            )

            result["financial_score"] = (
                float(financial_score)
            )

            result["lifecycle_score"] = (
                float(lifecycle_score)
            )

            result["purpose_score"] = (
                float(purpose_score)
            )

            result["transaction_score"] = (
                float(transaction_score)
            )

            result["final_score"] = (
                float(final_score)
            )

            ranked_results.append(
                result
            )

        # ----------------------------------------------------
        # Sort
        # ----------------------------------------------------

        ranked_results.sort(
            key=lambda x: x.get(
                "final_score",
                0.0
            ),
            reverse=True
        )

        # ----------------------------------------------------
        # Intent-specific ranking
        #
        # This is the important part.
        # ----------------------------------------------------

        if intent == "signature":
            ranked_results.sort(
                key=lambda x: (
                    x.get(
                        "signature_score",
                        0.0
                    ) * 0.60
                    + x.get(
                        "relevance_score",
                        0.0
                    ) * 0.25
                    + x.get(
                        "semantic_score",
                        0.0
                    ) * 0.15
                ),
                reverse=True
            )
        elif intent == "commercial":
            ranked_results.sort(
                key=lambda x: (
                    x.get(
                        "financial_score",
                        0.0
                    ) * 0.50
                    + x.get(
                        "relevance_score",
                        0.0
                    ) * 0.30
                    + x.get(
                        "semantic_score",
                        0.0
                    ) * 0.20
                ),
                reverse=True
            )

        # ----------------------------------------------------
        # Weak result filtering
        # ----------------------------------------------------

        filtered_results = []

        for result in ranked_results:

            relevance = result.get(
                "relevance_score",
                0.0
            )

            semantic = result.get(
                "semantic_score",
                0.0
            )

            lexical = result.get(
                "lexical_score",
                0.0
            )

            final_score = result.get(
                "final_score",
                0.0
            )

            document_score = result.get(
                "document_score",
                0.0
            )

            signature_score = result.get(
                "signature_score",
                0.0
            )

            financial_score = result.get(
                "financial_score",
                0.0
            )

            if intent == "signature":

                keep_result = (
                    signature_score >= 0.20
                    or relevance >= 0.15
                    or semantic >= 0.30
                    or document_score >= 0.50
                )

            elif intent == "commercial":

                keep_result = (
                    financial_score >= 0.20
                    or relevance >= 0.15
                    or semantic >= 0.30
                    or lexical >= 1.0
                    or document_score >= 0.50
                )

            else:

                keep_result = (
                    relevance >= 0.15
                    or semantic >= 0.30
                    or lexical >= 1.0
                    or final_score >= 0.30
                    or document_score >= 0.50
                )

            if keep_result:

                filtered_results.append(
                    result
                )

        if not filtered_results:

            filtered_results = (
                ranked_results[:top_k]
            )

        # ----------------------------------------------------
        # Document diversity
        #
        # Signature questions get more chunks because the
        # signature evidence may span several pages.
        # ----------------------------------------------------

        if intent == "signature":

            max_per_document = 4

        elif (
            has_explicit_document
            and intent == "commercial"
        ):

            max_per_document = 3

        else:

            max_per_document = 2

        final_results = (
            self._apply_document_diversity(
                filtered_results,
                top_k=top_k,
                max_per_document=max_per_document
            )
        )

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        logger.info(
            "Hybrid search completed | "
            "intent=%s | semantic=%s | lexical=%s | "
            "candidates=%s | final=%s",
            intent,
            len(semantic_results),
            len(lexical_results),
            len(ranked_results),
            len(final_results)
        )

        for rank, result in enumerate(
            final_results,
            start=1
        ):

            logger.info(
                "RANK=%s FILE=%s PAGE=%s CHUNK=%s "
                "FINAL=%.4f SEMANTIC=%.4f "
                "LEXICAL=%.4f RELEVANCE=%.4f "
                "DOC=%.4f INTENT=%.4f "
                "SIGNATURE=%.4f FINANCIAL=%.4f",
                rank,
                result["filename"],
                result.get(
                    "page_number"
                ),
                result["chunk_index"],
                result["final_score"],
                result["semantic_score"],
                result["lexical_score"],
                result["relevance_score"],
                result["document_score"],
                result["intent_score"],
                result["signature_score"],
                result["financial_score"],
            )

            logger.info(
                "========== RAG RETRIEVAL DEBUG =========="
            )

            logger.info(
                "QUERY=%s",
                query
            )

            logger.info(
                "INTENT=%s",
                intent
            )

            logger.info(
                "TARGET_DOCUMENTS=%s",
                list(target_documents)
            )

            for rank, result in enumerate(
                final_results,
                start=1,
            ):
                logger.info(
                    "RESULT %s | "
                    "FILE=%s | "
                    "PAGE=%s | "
                    "CHUNK=%s | "
                    "FINAL=%.4f | "
                    "SEMANTIC=%.4f | "
                    "LEXICAL=%.4f | "
                    "RELEVANCE=%.4f | "
                    "SIGNATURE=%.4f | "
                    "FINANCIAL=%.4f",
                    rank,
                    result.get("filename"),
                    result.get("page_number"),
                    result.get("chunk_index"),
                    result.get("final_score", 0.0),
                    result.get("semantic_score", 0.0),
                    result.get("lexical_score", 0.0),
                    result.get("relevance_score", 0.0),
                    result.get("signature_score", 0.0),
                    result.get("financial_score", 0.0),
                )

                logger.info(
                    "RESULT %s TEXT=%s",
                    rank,
                    result.get("text", "")[:1000],
                )

            logger.info(
                "========== END RAG RETRIEVAL DEBUG =========="
            )

        return final_results

    # ========================================================
    # DOCUMENT DIVERSITY
    # ========================================================

    def _apply_document_diversity(
        self,
        results,
        top_k=5,
        max_per_document=2,
    ):

        if not results:
            return []

        ranked_results = sorted(
            results,
            key=lambda item: (
                item.get(
                    "final_score",
                    0.0
                ),
                item.get(
                    "relevance_score",
                    0.0
                ),
                item.get(
                    "semantic_score",
                    0.0
                ),
                -item.get(
                    "chunk_index",
                    0
                ),
            ),
            reverse=True
        )

        selected = []
        document_counts = {}

        for result in ranked_results:

            filename = result.get(
                "filename",
                "unknown_document"
            )

            count = document_counts.get(
                filename,
                0
            )

            if count >= max_per_document:
                continue

            selected.append(
                result
            )

            document_counts[
                filename
            ] = count + 1

            if len(selected) >= top_k:
                break

        return selected

    # ========================================================
    # DELETE DOCUMENT
    # ========================================================

    def delete_document(
        self,
        document_id
    ):

        if not document_id:
            return 0

        with self.lock:

            vector_ids = []

            for vector_id, metadata in (
                self.metadata.items()
            ):

                if metadata.get(
                    "document_id"
                ) == document_id:

                    try:

                        vector_ids.append(
                            int(vector_id)
                        )

                    except (
                        TypeError,
                        ValueError
                    ):

                        logger.warning(
                            "Invalid vector ID: %s",
                            vector_id
                        )

            if not vector_ids:

                logger.info(
                    "No FAISS vectors found for document: %s",
                    document_id
                )

                return 0

            ids = np.asarray(
                vector_ids,
                dtype="int64"
            )

            removed = int(
                self.index.remove_ids(
                    ids
                )
            )

            if removed != len(
                vector_ids
            ):

                raise RuntimeError(
                    "FAISS deletion consistency check failed."
                )

            for vector_id in vector_ids:

                self.metadata.pop(
                    str(vector_id),
                    None
                )

            remaining = sum(
                1
                for metadata
                in self.metadata.values()
                if metadata.get(
                    "document_id"
                ) == document_id
            )

            if remaining != 0:

                raise RuntimeError(
                    "FAISS metadata deletion "
                    "consistency check failed."
                )

            self._save()

            logger.info(
                "Deleted %s vectors for document %s",
                removed,
                document_id
            )

            return removed

    # ========================================================
    # DOCUMENT VECTOR COUNT
    # ========================================================

    def get_document_vector_count(
        self,
        document_id
    ):

        return sum(
            1
            for metadata
            in self.metadata.values()
            if metadata.get(
                "document_id"
            ) == document_id
        )

    # ========================================================
    # TOTAL VECTOR COUNT
    # ========================================================

    def get_total_vector_count(self):

        return self.index.ntotal

    # ========================================================
    # CONSISTENCY
    # ========================================================

    def check_consistency(
        self,
        user_id=None
    ):

        with self.lock:

            if user_id is None:

                relevant_metadata = (
                    self.metadata.values()
                )

            else:

                relevant_metadata = [
                    metadata
                    for metadata
                    in self.metadata.values()
                    if metadata.get(
                        "user_id"
                    ) == user_id
                ]

            metadata_count = len(
                list(relevant_metadata)
            )

            faiss_count = int(
                self.index.ntotal
            )

            # A user-specific FAISS count cannot be obtained
            # directly from IndexIDMap2 without scanning IDs.
            # Therefore only perform exact count comparison
            # for the global consistency check.

            if user_id is None:

                consistent = (
                    faiss_count
                    == metadata_count
                )

            else:

                consistent = True

            document_counts = {}

            for metadata in (
                self.metadata.values()
            ):

                if user_id is not None:

                    if metadata.get(
                        "user_id"
                    ) != user_id:
                        continue

                document_id = metadata.get(
                    "document_id"
                )

                if not document_id:
                    continue

                document_counts[
                    document_id
                ] = (
                    document_counts.get(
                        document_id,
                        0
                    ) + 1
                )

            return {
                "consistent": consistent,
                "faiss_vectors": faiss_count,
                "metadata_entries": (
                    metadata_count
                ),
                "document_count": len(
                    document_counts
                ),
                "document_vector_counts":
                    document_counts,
            }

    def _lifecycle_score(
        self,
        query,
        text
    ):
        if (
            self._detect_intent(query)
            != "lifecycle"
        ):
            return 0.0
        
        if not text:
            return 0.0

        normalized = self._normalize_text(
            text
        )

        lifecycle_terms = {
            "effective",
            "commencement",
            "start",
            "expiry",
            "expire",
            "termination",
            "term",
            "tenure",
            "duration",
            "validity",
            "renewal",
            "renew",
            "period",
            "until",
        }

        matched = set(
            self._tokenize(
                normalized
            )
        ).intersection(
            lifecycle_terms
        )

        if not matched:
            return 0.0

        score = min(
            1.0,
            len(matched) * 0.15
        )

        return float(score)

    def _purpose_score(
        self,
        query,
        text
    ):
        if (
            self._detect_intent(query)
            != "purpose"
        ):
            return 0.0

        if not text:
            return 0.0

        normalized = self._normalize_text(
            text
        )

        purpose_terms = {
            "purpose",
            "objective",
            "business",
            "rationale",
            "need",
            "intent",
            "strategic",
            "commercial",
            "reason",
            "scope",
        }

        tokens = set(
            self._tokenize(
                normalized
            )
        )

        matched = tokens.intersection(
            purpose_terms
        )

        if not matched:
            return 0.0

        return float(
            min(
                1.0,
                len(matched) * 0.15
            )
        )

    def _transaction_score(
        self,
        query,
        text
    ):
        if (
            self._detect_intent(query)
            != "transaction"
        ):
            return 0.0

        if not text:
            return 0.0

        normalized = self._normalize_text(
            text
        )

        transaction_terms = {
            "purchase",
            "sale",
            "service",
            "services",
            "manufacturing",
            "royalty",
            "procurement",
            "design",
            "supply",
            "transportation",
        }

        tokens = set(
            self._tokenize(
                normalized
            )
        )

        matched = tokens.intersection(
            transaction_terms
        )

        if not matched:
            return 0.0

        return float(
            min(
                1.0,
                len(matched) * 0.15
            )
        )    

    def search_in_document(
        self,
        query,
        filename,
        top_k=5,
        user_id=None,
        resolved_filename=None,
    ):
        """
        Search ONLY inside one specific document.

        IMPORTANT:
        Unlike the previous implementation, this method does NOT
        perform a global FAISS search followed by document filtering.

        Instead:
            1. Identify vectors belonging to the requested document.
            2. Calculate semantic similarity only against those vectors.
            3. Calculate lexical relevance only against those vectors.
            4. Combine and rerank only those candidates.

        This guarantees that chunks from another document cannot
        influence document-restricted retrieval.
        """

        if not query or not query.strip():
            return []

        if not filename:
            return []

        filename = str(filename).strip()

        # ========================================================
        # FIND DOCUMENT VECTOR IDS
        # ========================================================

        allowed_vector_ids = []

        for vector_id, metadata in self.metadata.items():
            if user_id is not None:
                if metadata.get("user_id") != user_id:
                    continue

            metadata_filename = (
                metadata.get("filename", "")
            )

            if metadata_filename != filename:
                continue

            try:
                allowed_vector_ids.append(
                    int(vector_id)
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

        if not allowed_vector_ids:
            logger.warning(
                "No vectors found for document | "
                "filename=%s | user=%s",
                filename,
                user_id,
            )

            return []

        # ========================================================
        # EMBED QUERY
        # ========================================================

        query_embedding = (
            embedding_service.embed_text(query)
        )

        if query_embedding is None:
            return []

        query_vector = np.asarray(
            [query_embedding],
            dtype="float32",
        )

        if query_vector.ndim != 2:

            raise ValueError(
                "Query embedding must be a 2D vector."
            )

        if query_vector.shape[1] != self.dimension:

            raise ValueError(
                "Embedding dimension mismatch. "
                f"Expected={self.dimension}, "
                f"Received={query_vector.shape[1]}"
            )

        # ========================================================
        # DOCUMENT-ONLY SEMANTIC SEARCH
        # ========================================================
        #
        # IMPORTANT:
        # Do NOT call:
        #
        #     self.index.search(...)
        #
        # here.
        #
        # That searches the entire global index.
        #
        # Instead, reconstruct only vectors belonging to the
        # requested document and calculate their similarity.
        # ========================================================

        semantic_candidates = []

        for vector_id in allowed_vector_ids:
            try:

                document_vector = (
                    self.index.reconstruct(
                        int(vector_id)
                    )
                )

            except Exception as exc:
                logger.warning(
                    "Could not reconstruct FAISS vector | "
                    "vector_id=%s | filename=%s | error=%s",
                    vector_id,
                    filename,
                    exc,
                )

                continue

            document_vector = np.asarray(
                document_vector,
                dtype="float32",
            )

            if document_vector.shape[0] != self.dimension:
                continue

            semantic_score = float(
                np.dot(
                    query_vector[0],
                    document_vector,
                )
            )

            metadata = self.metadata.get(
                str(vector_id)
            )

            if not metadata:
                continue

            # Final safety check.
            if metadata.get("filename") != filename:
                continue

            if (
                user_id is not None
                and metadata.get("user_id") != user_id
            ):
                continue

            semantic_candidates.append(
                {
                    "score": semantic_score,
                    "semantic_score": semantic_score,
                    "lexical_score": 0.0,
                    "vector_id": vector_id,
                    "document_id":
                        metadata.get("document_id"),
                    "filename":
                        metadata.get(
                            "filename",
                            "",
                        ),
                    "chunk_index":
                        metadata.get(
                            "chunk_index",
                            0,
                        ),
                    "page_number":
                        metadata.get(
                            "page_number",
                            0,
                        ),
                    "text":
                        metadata.get(
                            "text",
                            "",
                        ),
                }
            )

        if not semantic_candidates:
            return []

        # ========================================================
        # KEEP A REASONABLE SEMANTIC CANDIDATE SET
        # ========================================================

        semantic_candidates.sort(
            key=lambda item: (
                item.get(
                    "semantic_score",
                    0.0,
                ),
                -item.get(
                    "chunk_index",
                    0,
                ),
            ),
            reverse=True,
        )

        candidate_limit = max(
            top_k * 10,
            50,
        )

        results = semantic_candidates[
            :candidate_limit
        ]

        # ========================================================
        # DOCUMENT-RESTRICTED LEXICAL SEARCH
        # ========================================================
        #
        # IMPORTANT:
        # The lexical search must also be restricted to this
        # document.
        #
        # We do NOT search all user documents and then take the
        # requested filename from the top 50.
        # ========================================================

        lexical_results = (
            self._lexical_search(
                query,
                top_k=candidate_limit,
                user_id=user_id,
                filename=filename,
            )
        )

        lexical_by_id = {
            item["vector_id"]: item
            for item in lexical_results
        }

        # ========================================================
        # NORMALIZE LEXICAL SCORE
        # ========================================================

        lexical_max = max(
            (
                float(
                    item.get(
                        "score",
                        0.0,
                    )
                )
                for item in lexical_results
            ),
            default=0.0,
        )

        for result in results:

            lexical = lexical_by_id.get(
                result["vector_id"]
            )

            if lexical:

                raw_lexical_score = float(
                    lexical.get(
                        "score",
                        0.0,
                    )
                )

                if lexical_max > 0:

                    result["lexical_score"] = (
                        raw_lexical_score
                        / lexical_max
                    )

                else:

                    result["lexical_score"] = 0.0

            else:

                result["lexical_score"] = 0.0

        # ========================================================
        # RE-RANK INSIDE DOCUMENT
        # ========================================================

        intent = self._detect_intent(
            query
        )

        for result in results:

            relevance_score = (
                self._query_relevance_score(
                    query,
                    result["text"],
                )
            )

            intent_score = (
                self._intent_score(
                    query,
                    result["text"],
                )
            )

            signature_score = (
                self._signature_score(
                    query,
                    result["text"],
                )
            )

            financial_score = (
                self._financial_score(
                    query,
                    result["text"],
                )
            )

            lifecycle_score = (
                self._lifecycle_score(
                    query,
                    result["text"],
                )
            )

            purpose_score = (
                self._purpose_score(
                    query,
                    result["text"],
                )
            )

            transaction_score = (
                self._transaction_score(
                    query,
                    result["text"],
                )
            )
 
            semantic_score = float(
                result.get(
                    "semantic_score",
                    0.0,
                )
            )

            lexical_score = float(
                result.get(
                    "lexical_score",
                    0.0,
                )
            )

            # ----------------------------------------------------
            # Store individual scores
            # ----------------------------------------------------

            result["relevance_score"] = float(
                relevance_score
            )

            result["intent_score"] = float(
                intent_score
            )

            result["signature_score"] = float(
                signature_score
            )

            result["financial_score"] = float(
                financial_score
            )

            result["lifecycle_score"] = float(
                lifecycle_score
            )

            result["purpose_score"] = float(
                purpose_score
            )

            result["transaction_score"] = float(
                transaction_score
            )

            # ----------------------------------------------------
            # Final score
            # ----------------------------------------------------
            # Scores are kept on comparable scales. The active
            # question intent receives an additional targeted bonus.
            # ----------------------------------------------------
            intent_bonus = 0.0

            if intent == "signature":
                intent_bonus = signature_score * 0.15
            elif intent == "commercial":
                intent_bonus = financial_score * 0.15
            elif intent == "lifecycle":
                intent_bonus = lifecycle_score * 0.15
            elif intent == "purpose":
                intent_bonus = purpose_score * 0.15
            elif intent == "transaction":
                intent_bonus = transaction_score * 0.15
            elif intent == "actualization":
                intent_bonus = intent_score * 0.15

            result["final_score"] = float(
                semantic_score * 0.40
                + lexical_score * 0.10
                + relevance_score * 0.25
                + intent_score * 0.05
                + intent_bonus
                + signature_score * 0.025
                + financial_score * 0.025
            )

        # ========================================================
        # INTENT-SPECIFIC RANKING
        # ========================================================

        if intent == "signature":

            results.sort(
                key=lambda item: (
                    item.get(
                        "signature_score",
                        0.0,
                    ) * 0.60
                    + item.get(
                        "relevance_score",
                        0.0,
                    ) * 0.25
                    + item.get(
                        "semantic_score",
                        0.0,
                    ) * 0.15
                ),
                reverse=True,
            )

        elif intent == "commercial":

            results.sort(
                key=lambda item: (
                    item.get(
                        "financial_score",
                        0.0,
                    ) * 0.50
                    + item.get(
                        "relevance_score",
                        0.0,
                    ) * 0.30
                    + item.get(
                        "semantic_score",
                        0.0,
                    ) * 0.20
                ),
                reverse=True,
            )

        elif intent in {
            "lifecycle",
            "purpose",
            "transaction",
            "actualization",
        }:

            score_key = {
                "lifecycle": "lifecycle_score",
                "purpose": "purpose_score",
                "transaction": "transaction_score",
            }.get(
                intent
            )

            if score_key:
                results.sort(
                    key=lambda item: (
                        item.get(
                            score_key,
                            0.0,
                        ) * 0.50
                        + item.get(
                            "relevance_score",
                            0.0,
                        ) * 0.30
                        + item.get(
                            "semantic_score",
                            0.0,
                        ) * 0.20
                    ),
                    reverse=True,
                )
            else:
                results.sort(
                    key=lambda item: (
                        item.get(
                            "intent_score",
                            0.0,
                        ) * 0.50
                        + item.get(
                            "relevance_score",
                            0.0,
                        ) * 0.30
                        + item.get(
                            "semantic_score",
                            0.0,
                        ) * 0.20
                    ),
                    reverse=True,
                )

        else:

            results.sort(
                key=lambda item: (
                    item.get(
                        "final_score",
                        0.0,
                    ),
                    item.get(
                        "relevance_score",
                        0.0,
                    ),
                    item.get(
                        "semantic_score",
                        0.0,
                    ),
                    -item.get(
                        "chunk_index",
                        0,
                    ),
                ),
                reverse=True,
            )

        # ========================================================
        # FINAL DOCUMENT SAFETY CHECK
        # ========================================================

        results = [
            result
            for result in results
            if result.get(
                "filename"
            ) == filename
            and (
                user_id is None
                or self.metadata.get(
                    str(result.get("vector_id")),
                    {}
                ).get("user_id") == user_id
            )
        ]

        # ========================================================
        # DEBUG LOGGING
        # ========================================================

        logger.info(
            "Document-restricted search completed | "
            "filename=%s | user=%s | "
            "query=%s | candidates=%s | results=%s",
            filename,
            user_id,
            query,
            len(allowed_vector_ids),
            len(results),
        )

        for rank, result in enumerate(
            results[:top_k],
            start=1,
        ):

            logger.info(
                "DOCUMENT RESULT %s | "
                "FILE=%s | PAGE=%s | "
                "CHUNK=%s | FINAL=%.4f | "
                "SEMANTIC=%.4f | "
                "LEXICAL=%.4f | "
                "RELEVANCE=%.4f | "
                "SIGNATURE=%.4f | "
                "FINANCIAL=%.4f",
                rank,
                result.get("filename"),
                result.get("page_number"),
                result.get("chunk_index"),
                result.get(
                    "final_score",
                    0.0,
                ),
                result.get(
                    "semantic_score",
                    0.0,
                ),
                result.get(
                    "lexical_score",
                    0.0,
                ),
                result.get(
                    "relevance_score",
                    0.0,
                ),
                result.get(
                    "signature_score",
                    0.0,
                ),
                result.get(
                    "financial_score",
                    0.0,
                ),
            )

        return results[:top_k]

    def resolve_document(
        self,
        query,
        user_id=None,
    ):
        """
        Resolve the most likely uploaded document from the query.
        Returns:
            {
                "filename": "...",
                "score": float,
                "ambiguous": bool,
                "candidates": [...]
            }
        Returns filename=None when there is no sufficiently strong
        document match.
        """
        if not query or not query.strip():
            return {
                "filename": None,
                "score": 0.0,
                "ambiguous": False,
                "candidates": [],
            }

        query_text = self._normalize_text(
            query
        )

        aliases = {
            "wh": "warehouse",
            "whs": "warehouse",
            "warehouses": "warehouse",
            "off": "office",
            "offs": "office",
            "offices": "office",
        }

        query_tokens = {
            aliases.get(
                token,
                token,
            )
            for token in query_text.split()
        }

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
            "please",
            "tell",
            "me",
            "give",
            "about",
            "agreement",
            "agreements",
            "document",
            "documents",
            "file",
            "files",
            "pdf",
            "contract",
            "contracts",
        }

        query_tokens -= generic_terms

        if not query_tokens:
            return {
                "filename": None,
                "score": 0.0,
                "ambiguous": False,
                "candidates": [],
            }

        # --------------------------------------------------------
        # Build unique documents
        # --------------------------------------------------------
        documents = {}

        for metadata in self.metadata.values():
            if user_id is not None:
                if metadata.get("user_id") != user_id:
                    continue

            filename = metadata.get(
                "filename",
                "",
            )

            if filename:
                documents[filename] = True

        candidates = []

        for filename in documents:
            filename_text = self._normalize_text(
                Path(filename).stem
            )

            filename_tokens = {
                aliases.get(
                    token,
                    token,
                )
                for token in filename_text.split()
            }

            filename_tokens -= generic_terms
            
            if not filename_tokens:
                continue

            overlap = (
                query_tokens
                & filename_tokens
            )

            if not overlap:
                continue

            # ----------------------------------------------------
            # Distinctive filename tokens are more important
            # than generic tokens such as warehouse / office.
            # ----------------------------------------------------
            distinctive = {
                token
                for token in filename_tokens
                if token not in {
                    "warehouse",
                    "office",
                    "space",
                    "lp",
                    "wh",
                }
                and not token.isdigit()
            }

            distinctive_overlap = (
                overlap
                & distinctive
            )

            query_coverage = (
                len(overlap)
                / len(query_tokens)
            )

            filename_coverage = (
                len(overlap)
                / len(filename_tokens)
            )

            score = (
                query_coverage * 5.0
                + filename_coverage * 3.0
                + len(distinctive_overlap) * 5.0
            )

            candidates.append(
                {
                    "filename": filename,
                    "score": score,
                    "overlap": sorted(overlap),
                    "distinctive_overlap":
                        sorted(
                            distinctive_overlap
                        ),
                }
            )

        candidates.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        if not candidates:
            return {
                "filename": None,
                "score": 0.0,
                "ambiguous": False,
                "candidates": [],
            }

        best = candidates[0]

        # --------------------------------------------------------
        # Ambiguity detection
        # --------------------------------------------------------
        ambiguous = False

        if len(candidates) > 1:
            second = candidates[1]

            if second["score"] >= (
                best["score"] * 0.90
            ):
                ambiguous = True

        # --------------------------------------------------------
        # Do not force a weak document match
        # --------------------------------------------------------
        if best["score"] < 3.0:
            return {
                "filename": None,
                "score": float(
                    best["score"]
                ),
                "ambiguous": False,
                "candidates": candidates[:5],
            }

        if ambiguous:
            logger.warning(
                "Ambiguous document resolution | "
                "query=%s | candidates=%s",
                query,
                candidates[:5],
            )

            return {
                "filename": None,
                "score": float(
                    best["score"]
                ),
                "ambiguous": True,
                "candidates": candidates[:5],
            }

        logger.info(
            "Document resolved | "
            "query=%s | filename=%s | score=%.4f",
            query,
            best["filename"],
            best["score"],
        )

        return {
            "filename": best["filename"],
            "score": float(
                best["score"]
            ),
            "ambiguous": False,
            "candidates": candidates[:5],
        }


# ============================================================
# GLOBAL VECTOR STORE
# ============================================================

vector_store = FAISSVectorStore()
