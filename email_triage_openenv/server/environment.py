"""Email Triage environment logic."""

import json
import uuid
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from openenv.core.env_server import Environment

from ..models import EmailAction, EmailObservation, EmailState

DATA_DIR = Path(__file__).parent / "data"

VALID_CATEGORIES = ["billing", "technical_support", "general_inquiry", "complaint", "feedback"]
VALID_PRIORITIES = ["low", "medium", "high", "urgent"]
VALID_DEPARTMENTS = ["finance", "engineering", "sales", "product", "support"]


class EmailTriageEnvironment(Environment):
    """OpenEnv-compliant email triage environment."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        # Load data
        with open(DATA_DIR / "emails.json") as f:
            self.emails = json.load(f)["emails"]

        with open(DATA_DIR / "ground_truth.json") as f:
            self.gt = json.load(f)["ground_truth"]

        # Episode state
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._current_index: int = 0
        self._task_level: int = 1
        self._total_score: float = 0.0
        self._completed_emails: list = []
        self._previous_actions: list = []

    async def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> EmailObservation:
        """Start a new episode."""
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._current_index = 0
        self._task_level = kwargs.get("task_level", 1)
        self._total_score = 0.0
        self._completed_emails = []
        self._previous_actions = []

        return self._get_observation()

    async def step(self, action: EmailAction, timeout_s: Optional[float] = None, **kwargs) -> EmailObservation:
        """Process an action and return next observation."""
        if self._current_index >= len(self.emails):
            obs = self._get_terminal_observation()
            obs.done = True
            obs.reward = 0.0
            return obs

        # Get current email and ground truth
        email = self.emails[self._current_index]
        gt = self.gt.get(email["email_id"], {})

        # Score the action
        score, breakdown = self._score_action(action, gt)

        # Update episode state
        self._total_score += score
        self._step_count += 1
        self._completed_emails.append(email["email_id"])
        self._previous_actions.append({
            "email_id": email["email_id"],
            "action": action.model_dump(),
            "score": score,
            "breakdown": breakdown
        })

        # Move to next email
        self._current_index += 1
        done = self._current_index >= len(self.emails)

        # Build observation
        if done:
            obs = self._get_terminal_observation()
            obs.reward = score
        else:
            obs = self._get_observation()
            obs.reward = score

        obs.done = done
        obs.feedback = self._build_feedback(action, gt, breakdown)
        obs.score_breakdown = breakdown
        obs.previous_actions = self._previous_actions[-5:]

        return obs

    def _score_action(self, action: EmailAction, gt: Dict) -> Tuple[float, Dict]:
        """Score action normalized to 0.0-1.0."""
        score = 0.0
        breakdown = {}

        # Task 1: category + priority (0.6 total)
        cat_correct = action.category == gt.get("category", "")
        pri_correct = action.priority == gt.get("priority", "")

        breakdown["category"] = {"got": action.category, "expected": gt.get("category", ""), "correct": cat_correct}
        breakdown["priority"] = {"got": action.priority, "expected": gt.get("priority", ""), "correct": pri_correct}

        score += (0.3 if cat_correct else 0)
        score += (0.3 if pri_correct else 0)

        # Task 2: department (0.2)
        if self._task_level >= 2 and "department" in gt:
            dept_correct = action.department == gt.get("department", "")
            breakdown["department"] = {"got": action.department, "expected": gt.get("department", ""),
                                       "correct": dept_correct}
            score += (0.2 if dept_correct else 0)

        # Task 3: escalation + reply (0.2 each)
        if self._task_level >= 3:
            if "needs_escalation" in gt:
                esc_correct = action.needs_escalation == gt.get("needs_escalation", False)
                breakdown["needs_escalation"] = {"got": action.needs_escalation,
                                                 "expected": gt.get("needs_escalation", False), "correct": esc_correct}
                score += (0.2 if esc_correct else 0)

            if action.reply_draft:
                draft_quality = self._score_reply_draft(action.reply_draft, gt)
                breakdown["reply_draft"] = {"quality": draft_quality}
                score += draft_quality * 0.2

        # Penalties
        if action.category and action.category not in VALID_CATEGORIES:
            score -= 0.1
            breakdown["invalid_category"] = {"penalty": -0.1}
        if action.priority and action.priority not in VALID_PRIORITIES:
            score -= 0.1
            breakdown["invalid_priority"] = {"penalty": -0.1}

        score = max(0.0, min(1.0, score))
        breakdown["total"] = score

        return score, breakdown

    def _score_reply_draft(self, draft: str, gt: Dict) -> float:
        """Score reply draft quality (0.0-1.0)."""
        if not draft:
            return 0.0

        score = 0.0
        professional_words = ["thank", "please", "apologize", "investigate", "resolve", "assist"]
        if any(word in draft.lower() for word in professional_words):
            score += 0.3

        actionable = ["will", "can", "let me", "i'll", "we'll"]
        if any(word in draft.lower() for word in actionable):
            score += 0.3

        words = len(draft.split())
        if 10 <= words <= 100:
            score += 0.2
        elif words > 0:
            score += 0.1

        if gt.get("needs_escalation", False):
            if any(word in draft.lower() for word in ["escalate", "manager", "supervisor"]):
                score += 0.2

        return min(1.0, score)

    def _build_feedback(self, action: EmailAction, gt: Dict, breakdown: Dict) -> str:
        """Build human-readable feedback."""
        parts = []
        for field, info in breakdown.items():
            if field == "total" or field in ["reply_draft", "invalid_category", "invalid_priority"]:
                continue
            if info.get("correct", False):
                parts.append(f"✓ {field}: '{info['got']}'")
            else:
                parts.append(f"✗ {field}: '{info['got']}' (expected '{info['expected']}')")

        if "reply_draft" in breakdown:
            parts.append(f"📝 reply: {breakdown['reply_draft']['quality']:.0%} quality")

        if "invalid_category" in breakdown:
            parts.append(f"⚠️ invalid category: '{action.category}'")
        if "invalid_priority" in breakdown:
            parts.append(f"⚠️ invalid priority: '{action.priority}'")

        parts.append(f"Score: {breakdown['total']:.2f}")
        return " | ".join(parts)

    def _get_observation(self) -> EmailObservation:
        """Get observation for current email."""
        if self._current_index >= len(self.emails):
            return self._get_terminal_observation()

        email = self.emails[self._current_index]

        return EmailObservation(
            done=False,
            reward=None,
            email_id=email["email_id"],
            subject=email["subject"],
            body=email["body"],
            sender=email["sender"],
            sender_name=email.get("sender_name", ""),
            timestamp=email["timestamp"],
            task_level=self._task_level,
            email_index=self._current_index,
            total_emails=len(self.emails),
            thread_history=self._get_thread_history(email["email_id"]),
            previous_actions=self._previous_actions[-5:],
            feedback=f"Process email {self._current_index + 1} of {len(self.emails)}",
            score_breakdown={}
        )

    def _get_terminal_observation(self) -> EmailObservation:
        """Get terminal observation when episode ends."""
        avg_score = self._total_score / self._step_count if self._step_count > 0 else 0.0

        return EmailObservation(
            done=True,
            reward=0.0,
            email_id="DONE",
            subject="Episode Complete",
            body=f"Processed {self._step_count} emails.\nTotal score: {self._total_score:.2f}\nAverage score: {avg_score:.2f}",
            sender="system@email-triage",
            sender_name="System",
            timestamp="",
            task_level=self._task_level,
            email_index=self._current_index,
            total_emails=len(self.emails),
            feedback=f"Final average: {avg_score:.2f}",
            score_breakdown={}
        )

    def _get_thread_history(self, email_id: str) -> list:
        """Get thread history for context."""
        email = next((e for e in self.emails if e["email_id"] == email_id), None)
        if not email:
            return []

        sender = email["sender"]
        return [
            e["subject"] for e in self.emails
            if e["sender"] == sender and e["email_id"] != email_id
        ][-3:]

    @property
    def state(self) -> EmailState:
        """Return current episode state."""
        avg_score = self._total_score / self._step_count if self._step_count > 0 else 0.0

        return EmailState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_level=self._task_level,
            current_email_index=self._current_index,
            total_emails=len(self.emails),
            total_score=self._total_score,
            average_score=avg_score,
            completed_emails=self._completed_emails,
            task_completed=self._current_index >= len(self.emails)
        )