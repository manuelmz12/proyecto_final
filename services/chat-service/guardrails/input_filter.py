import re
import logging

logger = logging.getLogger(__name__)

PROMPT_INJECTION_PATTERNS = [
    r"ignore.{0,30}instructions",
    r"disregard.{0,40}(instructions|told|said)",
    r"forget.{0,30}instructions",
    r"override.{0,30}instructions",
    r"you are now",
    r"act as (a |an )?(?!research|assistant|analyst)",
    r"jailbreak",
    r"dan mode",
    r"system prompt",
    r"reveal (your|the) (prompt|instructions|system)",
    r"new persona",
    r"pretend (you are|to be)",
]


def check_prompt_injection(text: str) -> bool:
    """Returns True if prompt injection is detected."""
    text_lower = text.lower()
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            logger.warning(f"Prompt injection detected: pattern='{pattern}'")
            return True
    return False


def validate_input(text: str) -> dict:
    """Validate user input. Returns dict with 'safe', 'text', 'reason'."""
    if not text or not text.strip():
        return {"safe": False, "text": text, "reason": "Empty input"}

    if len(text) > 2000:
        return {"safe": False, "text": text, "reason": "Input too long (max 2000 characters)"}

    if check_prompt_injection(text):
        return {"safe": False, "text": text, "reason": "Potential prompt injection detected"}

    return {"safe": True, "text": text, "reason": "OK"}
