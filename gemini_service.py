# gemini_service.py
# ============================================================
# gemini_service.py
# ============================================================

import os
import json
import re
import logging

#import google.generativeai as genai
from google import genai
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

logger = logging.getLogger(__name__)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not configured."
    )

# genai.configure(
#     api_key=GEMINI_API_KEY
# )

# model = genai.GenerativeModel(
#     "gemini-3.5-flash"
# )

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-3.5-flash"


# ============================================================
# CONSTANTS
# ============================================================

NO_INFORMATION_RESPONSE = (
    "I couldn't find that information in the uploaded document."
)


# ============================================================
# SIGNATURE QUESTION DETECTION
# ============================================================

def _is_signature_question(question):
    """
    Determine whether the user is asking about a
    signature/signatory.
    """

    if not question:
        return False

    normalized_question = (
        question.lower().strip()
    )

    signature_keywords = [
        "signatory",
        "signatories",
        "signature",
        "signatures",
        "signed",
        "signed by",
        "who signed",
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
        "digitally signed",
        "digital signature",
    ]

    return any(
        keyword in normalized_question
        for keyword in signature_keywords
    )


# ============================================================
# GEMINI ANSWER
# ============================================================

def ask_gemini(
    context,
    question,
    current_date=None,
):
    """
    Answer a question using ONLY the supplied document context.

    Important:
    - Gemini must not use outside knowledge.
    - Retrieval metadata is not factual evidence.
    - Empty/insufficient context must produce the standard
      fallback response.
    """

    context = (
        context
        if isinstance(context, str)
        else ""
    )

    question = (
        question
        if isinstance(question, str)
        else ""
    ).strip()

    if not question:
        return (
            "Please enter a question."
        )

    if not context.strip():
        return NO_INFORMATION_RESPONSE

    current_date = (
        str(current_date).strip()
        if current_date
        else "Not provided"
    )

    is_signature_question = (
        _is_signature_question(
            question
        )
    )

    # ========================================================
    # SIGNATURE PROMPT
    # ========================================================

    if is_signature_question:
        prompt = f"""
        You are an enterprise document question-answering assistant.        
        Your task is to answer the user's question using ONLY the
        supplied DOCUMENT CONTENT.
        ============================================================
        STRICT DOCUMENT-GROUNDING RULES
        ============================================================
        1. Do not use outside knowledge.
        2. Do not guess.
        3. Do not infer facts that are not explicitly supported by the
        supplied document content.
        4. Ignore irrelevant document content.
        5. Every factual statement in your answer must be directly
        supported by the supplied DOCUMENT CONTENT.
        6. If the supplied content does not contain enough information
        to answer the question, reply EXACTLY:
        "{NO_INFORMATION_RESPONSE}"
        7. Preserve exact values from the document when answering
        questions about:
        - amounts
        - rates
        - percentages
        - dates
        - durations
        - names
        - designations
        - clause numbers
        - agreement terms
        8. Do not silently correct spelling, numbers, dates, or names
        using outside knowledge.
        9. If the supplied document content contains conflicting values,
        do not select one based on outside knowledge.
        Instead clearly state that the supplied document content
        contains conflicting information and identify the conflicting
        values if they are available.
        10. Do not combine unrelated pieces of information.
        11. Do not infer relationships between:
        - people
        - companies
        - agreements
        - dates
        - amounts
        - clauses
        - locations
        unless the document explicitly establishes the relationship.
        12. Do not merge information from different documents unless the
        user explicitly asks for a comparison.
        13. If the user asks about a particular agreement/document and
        the supplied content does not clearly relate to that
        document, do not answer using another document.
        14. Be concise and direct.
        15. Do not mention:
        - FAISS
        - embeddings
        - retrieval
        - chunks
        - vector search
        - BM25
        - RRF
        - ranking
        - internal implementation
        16. Treat retrieval scores, page numbers, chunk numbers, document
        labels, and other metadata as navigation information only.
        They are not factual evidence.
        17. Do not answer a question merely because the retrieved content
        is topically related.
        18. The answer must be supported by actual DOCUMENT CONTENT text.
        19. If the supplied content contains only partial evidence, answer
        only the supported portion and explicitly state what is not
        specified.
        20. For questions asking "who", "which company", "what date",
        "what amount", "what rate", "what percentage", or "what
        clause", do not derive the answer from a name, number, or
        keyword appearing in an unrelated section.
        21. If the question asks for a calculated value and the document
        provides the required inputs, perform only the calculation
        necessary to answer the question. Do not introduce external
        assumptions.
        22. If a date, duration, amount, rate, or percentage must be
        calculated, clearly distinguish the document-stated value from
        the calculated value.
        ============================================================
        EVIDENCE SUFFICIENCY AND PARTIAL-EVIDENCE RULES
        ============================================================
        23. Distinguish carefully between:
        - information explicitly present in the supplied content
        - information not found in the supplied content
        - information that conflicts within the supplied content
        - information that cannot be determined from the supplied content
        24. "Not found in the supplied content" does NOT mean that the
        document itself definitely does not contain the information.
        Therefore, do NOT say:
        - "The document has no such clause."
        - "The document does not contain this information."
        unless the supplied content explicitly establishes that fact.
        Prefer:
        "The supplied document content does not specify this."
        or:
        "This could not be determined from the supplied document content."
        25. If only part of the requested information is supported,
        answer the supported portion and explicitly identify the
        unsupported portion.
        26. Never fill an evidence gap using:
        - general business knowledge
        - common contract practices
        - assumptions
        - patterns from other documents
        - information from another agreement
        - information from another document unless comparison was
            explicitly requested
        27. A retrieved passage being topically similar to the question
        is not sufficient evidence by itself.
        28. A name, amount, date, rate, company, designation, or clause
        appearing somewhere in the supplied content must not be used
        as the answer unless the surrounding content establishes that
        it answers the user's question.
        29. If the evidence supports only a general statement but not the
        specific detail requested, provide the general supported
        statement and clearly state that the requested specific detail
        is not specified.
        ============================================================
        MULTI-DOCUMENT AND COMPARISON RULES
        ============================================================
        Apply these rules whenever the user explicitly asks about,
        compares, contrasts, or requests information from multiple
        documents.
        30. Keep each document's evidence separate.
        31. Never transfer a fact from one document to another merely
        because the documents discuss the same subject.
        32. For every requested comparison topic, determine whether
        evidence is actually available for EACH document.
        33. If evidence is available for one document but not another,
        report the supported information for the first document and
        explicitly state that the corresponding information for the
        other document could not be determined from the supplied
        content.
        34. Do NOT convert missing evidence into a negative factual claim.
        Example:
        Correct:
        "AD Inbound.pdf specifies payment within 30 days.
            The supplied content for AD Outbound.pdf does not specify
            a payment period."
        Incorrect:
        "AD Inbound.pdf specifies payment within 30 days.
            AD Outbound.pdf has no payment terms."
        35. If the same topic has different values in different
        documents, present each document's value separately.
        36. Do not choose one document's value as the correct value for
        another document.
        37. If the documents contain conflicting information and the
        conflict is genuinely between the documents, clearly identify
        the values by document.
        38. If a comparison topic cannot be assessed because the supplied
        evidence is incomplete, explicitly say:
        "A complete comparison cannot be confirmed from the supplied
        document content."
        39. Do not claim that a clause is "missing" from a document merely
        because that clause was not present in the retrieved content.
        Instead say:
        "The supplied content does not specify this."
        40. If the user asks for a complete comparison across several
        topics, do not silently omit topics for which evidence is
        unavailable.
        Identify those topics as:
        "Not specified in the supplied content."
        41. When comparing documents, use the document name only to
        identify which supplied content the evidence came from.
        Do not treat the filename itself as factual evidence.
        42. If multiple documents are supplied but the user's question
        is clearly about only one of them, answer only from the
        relevant document.
        ============================================================
        COMPARISON RESPONSE STRUCTURE
        ============================================================
        For a comparison question, when appropriate, use this structure:
        <topic>
        <document 1>:
        <supported information or "Not specified in the supplied content.">
        <document 2>:
        <supported information or "Not specified in the supplied content.">
        Comparison:
        <concise comparison based only on the supplied evidence>
        If the evidence is insufficient for a complete comparison, state:
        "A complete comparison cannot be confirmed from the supplied
        document content."
        Do not manufacture a comparison merely because the documents
        share similar terminology.
        ============================================================
        CONFLICT HANDLING
        ============================================================
        43. If the supplied content contains multiple values for the
        same fact within the same document:
        - do not silently choose one
        - identify the conflicting values
        - explain that the supplied content is conflicting
        44. If the conflict is between two documents, identify each
        value together with its corresponding document.
        45. If the supplied content does not provide enough surrounding
        context to determine whether two values are actually
        conflicting, say that the relationship cannot be determined.
        ============================================================
        AGREEMENT ANALYSIS TASKS
        ============================================================
        Apply these rules only when the user requests the corresponding task:
        1. Executive summary:
        Organize the answer under Parties, Business purpose, Transaction
        type, Important dates, Commercial terms, Key obligations,
        Termination/renewal, Signature evidence, and Risks or missing
        information. Omit a heading only when no supporting content exists.
        2. Business rationale:
        Explain why the parties entered the agreement, the stated business
        need, commercial intent, and strategic benefit. Do not substitute
        the transaction category for the business rationale. If rationale
        is only implied by recitals or scope, label it as an interpretation
        supported by those passages rather than an explicit statement.
        3. Nature of transaction:
        Classify only from the supplied content, using a specific category
        such as sale of goods, purchase/procurement, service arrangement,
        design services, contract manufacturing, royalty/license, or other.
        Briefly identify the clause evidence supporting the classification.
        4. Commercial extraction:
        Separate pricing methodology, rates/amounts, mark-up or margin,
        payment timing, taxes, escalation/indexation, invoicing, and other
        commercial conditions. Preserve exact numbers and currencies.
        5. Agreement comparison:
        Keep each document separate. Present a compact comparison by topic,
        then list missing clauses, changed terms, inconsistencies, and newly
        introduced clauses.
        IMPORTANT:
        Do not call a clause missing merely because it was not found in
        the supplied retrieved content. Use "Not specified in the supplied
        content" unless the supplied content itself establishes absence.
        6. Transfer-pricing policy validation:
        Identify the agreement methodology and mark-up, identify the policy
        benchmark/range, compare them, and state Pass, Deviation, or Cannot
        determine. Never invent an approved range and never treat a general
        commercial clause as a TP-policy benchmark.
        7. Actualization:
        Identify the obligation, frequency, due date or trigger, responsible
        party, and required action. State which of these is not specified.
        ============================================================
        LIFECYCLE DATE HANDLING
        ============================================================
        The application provides the current system date below.
        For agreement lifecycle questions only, you may compare this
        system date with effective/start/end/expiry dates explicitly
        stated in the DOCUMENT CONTENT.
        Do not treat the system date as a date stated by the agreement.
        Do not invent an agreement date.
        If the document provides an expiry/end date and the system date
        is after it, the agreement is expired.
        If the system date falls within an explicitly stated validity
        period, the agreement is active.
        If the document does not provide enough dates, say that the
        status cannot be determined from the document.
        CURRENT SYSTEM DATE:
        {current_date}
        ============================================================
        IMPORTANT SOURCE RULE
        ============================================================
        The DOCUMENT CONTENT is the ONLY source of factual information.
        Document names, page numbers, retrieval scores, ranking scores,
        semantic scores, lexical scores, and other retrieval metadata
        are NOT factual evidence.
        ============================================================
        DOCUMENT CONTENT
        ============================================================
        {context}
        ============================================================
        USER QUESTION
        ============================================================
        {question}
        ============================================================
        ANSWER
        ============================================================
        """

    # ========================================================
    # GENERAL QUESTION PROMPT
    # ========================================================
    else:
        prompt = f"""
        ... your Phase 3 Step 4 GENERAL prompt ...
        """

    # ========================================================
    # LOG REQUEST
    # ========================================================

    logger.info(
        "========== GEMINI REQUEST =========="
    )

    logger.info(
        "QUESTION=%s",
        question
    )

    logger.info(
        "SIGNATURE_QUESTION=%s",
        is_signature_question
    )

    logger.info(
        "CONTEXT_LENGTH=%s",
        len(context)
    )

    logger.info(
        "PROMPT_LENGTH=%s",
        len(prompt)
    )

    logger.info(
        "========== END GEMINI REQUEST =========="
    )

    # ========================================================
    # CALL GEMINI
    # ========================================================

    try:

        # response = model.generate_content(
        #     prompt
        # )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        response_text = (
            getattr(
                response,
                "text",
                ""
            )
            or ""
        ).strip()

        if not response_text:
            logger.warning(
                "Gemini returned an empty response."
            )

            return NO_INFORMATION_RESPONSE

        logger.info(
            "========== GEMINI RESPONSE =========="
        )

        logger.info(
            "GEMINI RESPONSE=%s",
            response_text
        )

        logger.info(
            "========== END GEMINI RESPONSE =========="
        )

        return response_text

    except Exception as exc:

        logger.exception(
            "Gemini answer generation failed: %s",
            exc
        )

        raise


# ============================================================
# FOLLOW-UP QUESTION REWRITER
# ============================================================

def rewrite_followup_question(
    question,
    last_question=None,
    last_answer=None,
    current_documents=None,
    explicit_document=None,
    current_document=None,
):
    """
    Rewrite a conversational follow-up question into a
    standalone retrieval question.

    This function does NOT answer the question.

    Example:

        Previous:
        "What is the rent of the Inbound warehouse?"

        Current:
        "What about the outbound one?"

        Result:
        "What is the rent of the Outbound warehouse?"

    The function fails safely and returns the original question
    whenever rewriting cannot be performed safely.
    """

    # ========================================================
    # EMPTY QUESTION
    # ========================================================

    if not question or not question.strip():

        return {
            "is_follow_up": False,
            "standalone_question":
                question or "",
            "target_document":
                explicit_document,
        }

    current_question = (
        question.strip()
    )

    previous_question = (
        last_question.strip()
        if isinstance(
            last_question,
            str
        )
        else ""
    )

    previous_answer = (
        last_answer.strip()
        if isinstance(
            last_answer,
            str
        )
        else ""
    )

    documents = (
        current_documents
        if isinstance(
            current_documents,
            list
        )
        else []
    )

    # ========================================================
    # NO PREVIOUS CONTEXT
    # ========================================================

    if (
        not previous_question
        and not previous_answer
    ):

        return {
            "is_follow_up": False,
            "standalone_question":
                current_question,
            "target_document":
                explicit_document,
        }

    # ========================================================
    # LIMIT PREVIOUS ANSWER
    # ========================================================

    previous_answer_context = (
        previous_answer[:4000]
        if previous_answer
        else ""
    )

    # ========================================================
    # DOCUMENT LIST
    # ========================================================

    document_list = "\n".join(
        f"- {filename}"
        for filename in documents
        if filename
    )

    if not document_list:
        document_list = (
            "- No document list available"
        )

    # ========================================================
    # DOCUMENT VALUES
    # ========================================================

    resolved_document = (
        explicit_document
        if explicit_document
        else "None"
    )

    active_document = (
        current_document
        if current_document
        else "None"
    )

    # ========================================================
    # REWRITING PROMPT
    # ========================================================

    prompt = f"""
You are a query-rewriting component inside an enterprise
document RAG system.

Your ONLY job is to determine whether the CURRENT USER QUESTION
depends on the previous conversation and, if necessary, rewrite
it into a standalone retrieval question.

DO NOT answer the question.

DO NOT summarize the documents.

DO NOT invent facts.

DO NOT add information that is not supported by the conversation.

============================================================
PREVIOUS USER QUESTION
============================================================

{previous_question}

============================================================
PREVIOUS ANSWER
============================================================

{previous_answer_context}

============================================================
CURRENT USER QUESTION
============================================================

{current_question}

============================================================
AVAILABLE UPLOADED DOCUMENTS
============================================================

{document_list}

============================================================
DETERMINISTICALLY RESOLVED DOCUMENT
============================================================

{resolved_document}

============================================================
CURRENT ACTIVE DOCUMENT
============================================================

{active_document}

============================================================
RULES
============================================================

1. If the current question is already standalone, keep it
   essentially unchanged.

2. If the current question is a follow-up, resolve references
   such as:

   - it
   - its
   - this
   - that
   - this one
   - that one
   - the other one
   - the second one
   - the first one
   - outbound one
   - inbound one
   - the same
   - and the rate
   - what about renewal
   - the duration
   - who signed it

3. DOCUMENT PRIORITY:

   If a document has been deterministically resolved from the
   CURRENT USER QUESTION, that document is authoritative.

   NEVER replace an explicitly resolved current document with
   the previous/current active document.

   Example:

   Current question:
   "What about AD Outbound?"

   Deterministically resolved document:
   "AD Outbound.pdf"

   Previous active document:
   "AD Inbound.pdf"

   The target document MUST remain:
   "AD Outbound.pdf"

4. If there is no explicit document in the current question,
   use CURRENT ACTIVE DOCUMENT when resolving:

   - it
   - this
   - that
   - the document
   - the agreement
   - its
   - who signed it

5. Only use another document when the current question explicitly
   refers to another document or the conversation clearly
   establishes that another document is intended.

6. Preserve the user's requested subject.

   Example:

   Previous:
   "What is the rent of the Inbound warehouse?"

   Current:
   "What about the outbound one?"

   Rewrite:
   "What is the rent of the Outbound warehouse?"

7. If the current question is:

   "And the rate?"

   and the previous question was about the rent of the
   Inbound warehouse, preserve the subject and document.

8. Do not invent a document name.

9. Do not invent a business term, amount, date, person,
   location, or other factual detail.

10. If the relationship cannot be safely determined:

    return the current question unchanged and set
    "is_follow_up" to false.

11. Return ONLY valid JSON.
11.5. Do not wrap the JSON in markdown.
Return raw JSON only.
Do not include:
- ```json
- ```
- explanations
- comments
- additional keys
- text before the JSON
- text after the JSON

12. Use EXACTLY this structure:

{{
    "is_follow_up": true,
    "standalone_question": "rewritten question",
    "target_document": "document filename or null"
}}

============================================================
"""

    # ========================================================
    # LOG
    # ========================================================

    logger.info(
        "========== FOLLOW-UP REWRITE REQUEST =========="
    )

    logger.info(
        "CURRENT QUESTION=%s",
        current_question
    )

    logger.info(
        "PREVIOUS QUESTION=%s",
        previous_question
    )

    logger.info(
        "EXPLICIT DOCUMENT=%s",
        explicit_document
    )

    logger.info(
        "CURRENT DOCUMENT=%s",
        current_document
    )

    # ========================================================
    # CALL GEMINI
    # ========================================================

    try:

        # response = model.generate_content(
        #     prompt
        # )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        response_text = (
            getattr(
                response,
                "text",
                ""
            )
            or ""
        ).strip()

        if not response_text:

            logger.warning(
                "Follow-up rewriter returned empty response."
            )

            return {
                "is_follow_up": False,
                "standalone_question":
                    current_question,
                "target_document":
                    explicit_document,
            }

        # ====================================================
        # REMOVE MARKDOWN JSON FENCE
        # ====================================================

        response_text = re.sub(
            r"^```(?:json)?\s*",
            "",
            response_text,
            flags=re.IGNORECASE,
        )

        response_text = re.sub(
            r"\s*```$",
            "",
            response_text,
        ).strip()

        # ====================================================
        # PARSE JSON
        # ====================================================

        parsed = json.loads(
            response_text
        )

        is_follow_up = bool(
            parsed.get(
                "is_follow_up",
                False
            )
        )

        standalone_question = (
            parsed.get(
                "standalone_question"
            )
        )

        target_document = (
            parsed.get(
                "target_document"
            )
        )

        # ====================================================
        # VALIDATE QUESTION
        # ====================================================

        if not isinstance(
            standalone_question,
            str
        ):
            standalone_question = (
                current_question
            )

        standalone_question = (
            standalone_question.strip()
        )

        if not standalone_question:
            standalone_question = (
                current_question
            )

        # ====================================================
        # VALIDATE DOCUMENT
        # ====================================================

        if target_document:

            target_document = str(
                target_document
            ).strip()

            # Gemini is NEVER allowed to invent a filename.

            if (
                target_document
                not in documents
            ):

                logger.warning(
                    "Follow-up rewriter returned "
                    "unknown document: %s",
                    target_document,
                )

                target_document = (
                    explicit_document
                )

        else:

            target_document = (
                explicit_document
            )

        # ====================================================
        # SAFETY RULE
        #
        # Explicit document from the current question always
        # wins over Gemini's returned document.
        # ====================================================

        if explicit_document:

            target_document = (
                explicit_document
            )

        # ====================================================
        # LOG RESULT
        # ====================================================

        logger.info(
            "Follow-up rewrite completed | "
            "original=%s | "
            "follow_up=%s | "
            "rewritten=%s | "
            "target_document=%s",
            current_question,
            is_follow_up,
            standalone_question,
            target_document,
        )

        return {
            "is_follow_up":
                is_follow_up,

            "standalone_question":
                standalone_question,

            "target_document":
                target_document,
        }

    except json.JSONDecodeError as exc:

        logger.warning(
            "Follow-up rewriter returned invalid JSON: %s",
            exc,
        )

        return {
            "is_follow_up": False,
            "standalone_question":
                current_question,
            "target_document":
                explicit_document,
        }

    except Exception as exc:

        logger.exception(
            "Follow-up question rewriting failed: %s",
            exc,
        )

        # ====================================================
        # FAIL SAFE
        # ====================================================

        return {
            "is_follow_up": False,
            "standalone_question":
                current_question,
            "target_document":
                explicit_document,
        }
