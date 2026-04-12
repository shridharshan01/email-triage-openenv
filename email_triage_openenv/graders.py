"""
Three programmatic graders for email triage tasks.
Each returns a score strictly in (0, 1) — never 0.0 or 1.0.
"""

from typing import Dict, Any

try:
    from .models import EmailAction
except ImportError:
    EmailAction = None

MIN_SCORE = 0.01
MAX_SCORE = 0.99


def grade_task_easy(action, ground_truth: Dict[str, Any]) -> float:
    score = 0.0

    if action.category == ground_truth.get("category", ""):
        score += 0.499
    if action.priority == ground_truth.get("priority", ""):
        score += 0.499

    return max(MIN_SCORE, min(MAX_SCORE, score))


def grade_task_medium(action, ground_truth: Dict[str, Any]) -> float:
    score = 0.0

    if action.category == ground_truth.get("category", ""):
        score += 0.399
    if action.priority == ground_truth.get("priority", ""):
        score += 0.299
    if action.department == ground_truth.get("department", ""):
        score += 0.299

    return max(MIN_SCORE, min(MAX_SCORE, score))


def grade_task_hard(action, ground_truth: Dict[str, Any]) -> float:
    score = 0.0

    if action.category == ground_truth.get("category", ""):
        score += 0.249
    if action.priority == ground_truth.get("priority", ""):
        score += 0.199
    if action.department == ground_truth.get("department", ""):
        score += 0.199
    if action.needs_escalation == ground_truth.get("needs_escalation", False):
        score += 0.149

    if action.reply_draft:
        draft_score = _score_reply_quality(action.reply_draft, ground_truth)
        score += draft_score * 0.2

    return max(MIN_SCORE, min(MAX_SCORE, score))


def _score_reply_quality(draft: str, ground_truth: Dict[str, Any]) -> float:
    quality = 0.0

    professional = ["thank", "please", "apologize", "investigate", "resolve"]
    if any(word in draft.lower() for word in professional):
        quality += 0.3

    actionable = ["will", "can", "let me", "i'll", "we'll"]
    if any(word in draft.lower() for word in actionable):
        quality += 0.3

    words = len(draft.split())
    if 10 <= words <= 100:
        quality += 0.2
    elif words > 0:
        quality += 0.1

    if ground_truth.get("needs_escalation", False):
        if any(word in draft.lower() for word in ["escalate", "manager", "supervisor"]):
            quality += 0.2

    return max(MIN_SCORE, min(MAX_SCORE, quality))
