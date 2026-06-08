import re

def redact_pii(text: str) -> str:
    if not text:
        return text

    PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",

    "phone": r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b",

    "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",

    "pan": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",

    "passport": r"\b[A-Z][0-9]{7}\b",

    "driving_license": r"\b[A-Z]{2}[0-9]{2}[0-9]{11,13}\b",

    "bank_account": r"\b\d{9,18}\b",

    "ifsc": r"\b[A-Z]{4}0[A-Z0-9]{6}\b",

    "card_number": r"\b(?:\d[ -]*?){13,19}\b",

    "cvv": r"\b\d{3,4}\b",

    "upi": r"\b[a-zA-Z0-9._-]+@[a-zA-Z]+\b",

    "github_token": r"\bgh[pousr]_[A-Za-z0-9_-]+\b",

    "openai_key": r"\bsk-[A-Za-z0-9_-]+\b",

    "jwt": r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b",

    "ip_v6": r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b",
    
    "ip_v4": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}

    for pattern in PATTERNS.values():
        text = re.sub(pattern, "[REDACTED]", text)

    return text