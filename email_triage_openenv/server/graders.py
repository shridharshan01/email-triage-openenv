"""
Three programmatic graders for email triage tasks.
Each returns a score in (0, 1) - strictly between 0 and 1.
"""

from typing import Dict, Any
from .models import EmailAction

# Constants for boundary avoidance
MIN_SCORE = 0.001
MAX_SCORE = 0.999


def grade_task_easy(action: EmailAction, ground_truth: Dict[str, Any]) -> float:
    """
    Easy task: Classify category and priority only.
    Returns score between MIN_SCORE and MAX_SCORE
    """
    score = 0.0

    if action.category == ground_truth.get("category", ""):
        score += 0.5
    if action.priority == ground_truth.get("priority", ""):
        score += 0.5

    # Avoid boundaries
    if score <= 0.0:
        score = MIN_SCORE
    elif score >= 1.0:
        score = MAX_SCORE

    return max(MIN_SCORE, min(MAX_SCORE, score))


def grade_task_medium(action: EmailAction, ground_truth: Dict[str, Any]) -> float:
    """
    Medium task: Category + priority + department.
    Returns score between MIN_SCORE and MAX_SCORE
    """
    score = 0.0

    if action.category == ground_truth.get("category", ""):
        score += 0.4
    if action.priority == ground_truth.get("priority", ""):
        score += 0.3
    if action.department == ground_truth.get("department", ""):
        score += 0.3

    if score <= 0.0:
        score = MIN_SCORE
    elif score >= 1.0:
        score = MAX_SCORE

    return max(MIN_SCORE, min(MAX_SCORE, score))


def grade_task_hard(action: EmailAction, ground_truth: Dict[str, Any]) -> float:
    """
    Hard task: Category + priority + department + escalation + reply quality.
    Returns score between MIN_SCORE and MAX_SCORE
    """
    score = 0.0

    if action.category == ground_truth.get("category", ""):
        score += 0.25
    if action.priority == ground_truth.get("priority", ""):
        score += 0.2
    if action.department == ground_truth.get("department", ""):
        score += 0.2
    if action.needs_escalation == ground_truth.get("needs_escalation", False):
        score += 0.15

    if action.reply_draft:
        draft_score = _score_reply_quality(action.reply_draft, ground_truth)
        score += draft_score * 0.2

    if score <= 0.0:
        score = MIN_SCORE
    elif score >= 1.0:
        score = MAX_SCORE

    return max(MIN_SCORE, min(MAX_SCORE, score))


def _score_reply_quality(draft: str, ground_truth: Dict[str, Any]) -> float:
    """Score reply draft quality between MIN_SCORE and MAX_SCORE."""
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

    if quality <= 0.0:
        quality = MIN_SCORE
    elif quality >= 1.0:
        quality = MAX_SCORE

    return max(MIN_SCORE, min(MAX_SCORE, quality))