import re
import logging
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

logger = logging.getLogger(__name__)

_analyzer: AnalyzerEngine | None = None

PROMPT_INJECTION_PATTERNS = [
    r"ignore (all |previous |above )?instructions",
    r"disregard (all |previous |above )?instructions",
    r"you are now",
    r"act as (a |an )?(?!research|assistant|analyst)",
    r"forget (everything|your instructions)",
    r"jailbreak",
    r"DAN mode",
    r"system prompt",
    r"reveal (your|the) (prompt|instructions|system)",
]

PII_ENTITIES = ["PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD", "IBAN_CODE", "NRP"]


def _get_analyzer() -> AnalyzerEngine:
    global _analyzer
    if _analyzer is None:
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        })
        _analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
    return _analyzer


def check_prompt_injection(text: str) -> bool:
    """Returns True if prompt injection is detected."""
    text_lower = text.lower()
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            logger.warning(f"Prompt injection detected: pattern='{pattern}'")
            return True
    return False


def redact_pii(text: str) -> str:
    """Replace PII with [REDACTED_TYPE] placeholders."""
    try:
        analyzer = _get_analyzer()
        results = analyzer.analyze(text=text, entities=PII_ENTITIES, language="en")
        if not results:
            return text
        redacted = text
        for result in sorted(results, key=lambda r: r.start, reverse=True):
            placeholder = f"[REDACTED_{result.entity_type}]"
            redacted = redacted[: result.start] + placeholder + redacted[result.end :]
        return redacted
    except Exception as e:
        logger.warning(f"PII redaction failed: {e}")
        return text


def validate_input(text: str) -> dict:
    """Validate and sanitize user input. Returns dict with 'safe', 'text', 'reason'."""
    if not text or not text.strip():
        return {"safe": False, "text": text, "reason": "Empty input"}

    if len(text) > 2000:
        return {"safe": False, "text": text, "reason": "Input too long (max 2000 characters)"}

    if check_prompt_injection(text):
        return {"safe": False, "text": text, "reason": "Potential prompt injection detected"}

    sanitized = redact_pii(text)
    return {"safe": True, "text": sanitized, "reason": "OK"}
