import calendar
import re
from datetime import date, timedelta

from services.agreement_expiry_service import (
    DATE_PATTERN,
    MONTHS,
    extract_agreement_expiry,
)


START_DATE_PATTERNS = [
    re.compile(
        rf"(?:effective|commencing|commences|valid)\s+(?:as\s+of\s+|from\s+|on\s+){DATE_PATTERN}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:effective|commencement|start)\s+date\s*(?:is|:)?\s*{DATE_PATTERN}",
        re.IGNORECASE,
    ),
]

TENURE_PATTERN = re.compile(
    r"valid\s+for\s+(?:(?:\w+\s+)?\((?P<number>\d+)\)|(?P<plain>\d+))\s*"
    r"(?P<unit>years?|months?)",
    re.IGNORECASE,
)

ESIGN_PATTERN = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z .'-]{2,80})\s*\n\s*(?P=name)\s*\n"
    r".{0,120}?(?:e\s*[- ]?sign\s+by|signdesk)",
    re.IGNORECASE | re.DOTALL,
)

TRANSACTION_RULES = [
    ("Contract Manufacturing", ("contract manufacturing", "manufacturing services")),
    ("Design Services", ("design services", "engineering design", "design and development")),
    ("Royalty", ("royalty", "license fee", "licence fee")),
    ("Sale of Goods", ("sale of goods", "sell goods", "seller", "buyer")),
    ("Purchase / Procurement", ("procurement services", "purchase agreement", "procure on behalf")),
    (
        "Service Arrangement",
        (
            "service provider",
            "scope of services",
            "provide the services",
            "transportation services",
            "logistics services",
        ),
    ),
]


def _date_from_match(match):
    try:
        return date(
            int(match.group("year")),
            MONTHS[match.group("month").lower()],
            int(match.group("day")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _add_months(start_date, months):
    month_index = start_date.month - 1 + months
    year = start_date.year + month_index // 12
    month = month_index % 12 + 1
    day = min(start_date.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _extract_start_date(text):
    candidates = []
    for pattern in START_DATE_PATTERNS:
        for match in pattern.finditer(text):
            parsed_date = _date_from_match(match)
            if parsed_date:
                candidates.append(parsed_date)
    return min(candidates) if candidates else None


def _extract_tenure_months(text):
    match = TENURE_PATTERN.search(text)
    if not match:
        return None

    value = int(match.group("number") or match.group("plain"))
    return value * 12 if match.group("unit").lower().startswith("year") else value


def _extract_signatories(text):
    signatories = []
    seen = set()

    for match in ESIGN_PATTERN.finditer(text):
        name = re.sub(r"\s+", " ", match.group("name")).strip()
        normalized_name = name.casefold()
        if normalized_name not in seen:
            seen.add(normalized_name)
            signatories.append(name)

    return signatories


def _classify_transaction(text):
    normalized = text.lower()
    best_category = None
    best_score = 0

    for category, terms in TRANSACTION_RULES:
        score = sum(normalized.count(term) for term in terms)
        if score > best_score:
            best_category = category
            best_score = score

    return best_category


def analyze_agreement(text, current_date=None, expiry_override=None):
    normalized_text = re.sub(r"[ \t]+", " ", str(text or ""))
    today = current_date or date.today()
    start_date = _extract_start_date(normalized_text)
    tenure_months = _extract_tenure_months(normalized_text)
    expiry_value = expiry_override or extract_agreement_expiry(normalized_text)
    end_date = date.fromisoformat(expiry_value) if expiry_value else None
    end_date_source = "Document" if end_date else None

    if not end_date and start_date and tenure_months:
        end_date = _add_months(start_date, tenure_months) - timedelta(days=1)
        expiry_value = end_date.isoformat()
        end_date_source = "Calculated from start date and tenure"

    if end_date:
        lifecycle_status = "Expired" if end_date < today else "Active"
    elif start_date and start_date > today:
        lifecycle_status = "Not Started"
    else:
        lifecycle_status = "Unknown"

    signatories = _extract_signatories(normalized_text)
    if len(signatories) >= 2:
        signature_status = "Multiple Signatures Detected"
    elif len(signatories) == 1:
        signature_status = "Potentially Incomplete"
    else:
        signature_status = "Not Detected"

    actualization_required = bool(
        re.search(
            r"\bactuali[sz](?:ation|e|ed|ing)\b",
            normalized_text,
            re.IGNORECASE,
        )
    )

    return {
        "start_date": start_date.isoformat() if start_date else None,
        "tenure_months": tenure_months,
        "expiry_date": expiry_value,
        "end_date_source": end_date_source,
        "lifecycle_status": lifecycle_status,
        "signatories": signatories,
        "signatory_count": len(signatories),
        "signature_status": signature_status,
        "signature_complete": None,
        "transaction_type": _classify_transaction(normalized_text),
        "actualization_required": actualization_required,
    }
