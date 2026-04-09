"""
Three programmatic graders for email triage tasks.
Each returns a score in (0, 1) - strictly between 0 and 1.
"""

from typing import Dict, Any
from .models import EmailAction


def grade_task_easy(action: EmailAction, ground_truth: Dict[str, Any]) -> float:
    """
    Easy task: Classify category and priority only.
    Max score: 1.0 (0.5 category + 0.5 priority)
    Returns score between 0.001 and 0.999
    """
    score = 0.0

    # Category classification (0.5)
    if action.category == ground_truth.get("category", ""):
        score += 0.5

    # Priority classification (0.5)
    if action.priority == ground_truth.get("priority", ""):
        score += 0.5

    # Avoid boundaries (0.0 and 1.0)
    if score <= 0.0:
        score = 0.001
    elif score >= 1.0:
        score = 0.999

    return score


def grade_task_medium(action: EmailAction, ground_truth: Dict[str, Any]) -> float:
    """
    Medium task: Category + priority + department.
    Max score: 1.0 (0.4 category + 0.3 priority + 0.3 department)
    Returns score between 0.001 and 0.999
    """
    score = 0.0

    # Category (0.4)
    if action.category == ground_truth.get("category", ""):
        score += 0.4

    # Priority (0.3)
    if action.priority == ground_truth.get("priority", ""):
        score += 0.3

    # Department (0.3)
    if action.department == ground_truth.get("department", ""):
        score += 0.3

    # Avoid boundaries (0.0 and 1.0)
    if score <= 0.0:
        score = 0.001
    elif score >= 1.0:
        score = 0.999

    return score


def grade_task_hard(action: EmailAction, ground_truth: Dict[str, Any]) -> float:
    """
    Hard task: Category + priority + department + escalation + reply quality.
    Max score: 1.0 (0.25 category + 0.2 priority + 0.2 department + 0.15 escalation + 0.2 reply)
    Returns score between 0.001 and 0.999
    """
    score = 0.0

    # Category (0.25)
    if action.category == ground_truth.get("category", ""):
        score += 0.25

    # Priority (0.2)
    if action.priority == ground_truth.get("priority", ""):
        score += 0.2

    # Department (0.2)
    if action.department == ground_truth.get("department", ""):
        score += 0.2

    # Escalation (0.15)
    if action.needs_escalation == ground_truth.get("needs_escalation", False):
        score += 0.15

    # Reply draft quality (0.2)
    if action.reply_draft:
        draft_score = _score_reply_quality(action.reply_draft, ground_truth)
        score += draft_score * 0.2

    # Avoid boundaries (0.0 and 1.0)
    if score <= 0.0:
        score = 0.001
    elif score >= 1.0:
        score = 0.999

    return min(0.999, max(0.001, score))


def _score_reply_quality(draft: str, ground_truth: Dict[str, Any]) -> float:
    """
    Score reply draft quality.
    Returns score strictly between 0.001 and 0.999
    """
    quality = 0.0

    # Professional tone (0.3)
    professional = ["thank", "please", "apologize", "investigate", "resolve"]
    if any(word in draft.lower() for word in professional):
        quality += 0.3

    # Actionable content (0.3)
    actionable = ["will", "can", "let me", "i'll", "we'll"]
    if any(word in draft.lower() for word in actionable):
        quality += 0.3

    # Appropriate length (0.2)
    words = len(draft.split())
    if 10 <= words <= 100:
        quality += 0.2
    elif words > 0:
        quality += 0.1

    # Escalation mention if needed (0.2)
    if ground_truth.get("needs_escalation", False):
        if any(word in draft.lower() for word in ["escalate", "manager", "supervisor"]):
            quality += 0.2

    # Avoid boundaries (0.0 and 1.0)
    if quality <= 0.0:
        quality = 0.001
    elif quality >= 1.0:
        quality = 0.999

    return min(0.999, max(0.001, quality))