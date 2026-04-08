from typing import Optional, List, Dict, Any
from pydantic import Field
from dataclasses import dataclass
from openenv.core.env_server import Action, Observation, State


class EmailAction(Action):
    """Action space for email triage."""
    category: str = Field(default="",
                          description="Email category: billing, technical_support, general_inquiry, complaint, feedback")
    priority: str = Field(default="", description="Priority: low, medium, high, urgent")
    department: str = Field(default="", description="Department: finance, engineering, sales, product, support")
    reply_draft: str = Field(default="", description="Draft reply text (optional)")
    needs_escalation: bool = Field(default=False, description="Whether this needs manager escalation")
    is_duplicate: bool = Field(default=False, description="Whether this is a duplicate email")


class EmailObservation(Observation):
    """Observation space - what the agent sees."""
    # Inherited: done: bool, reward: Optional[float]

    # Email content
    email_id: str
    subject: str
    body: str
    sender: str
    sender_name: str = ""
    timestamp: str

    # Task metadata
    task_level: int = Field(ge=1, le=3, description="Difficulty level 1=easy, 2=medium, 3=hard")
    email_index: int = 0
    total_emails: int = 0

    # Thread context (for medium/hard tasks)
    thread_history: List[str] = Field(default_factory=list)
    previous_actions: List[Dict[str, Any]] = Field(default_factory=list)

    # Feedback
    feedback: str = ""
    score_breakdown: Dict[str, Any] = Field(default_factory=dict)


class EmailState(State):
    """Episode state tracking."""
    # Inherited: episode_id: Optional[str], step_count: int

    task_level: int = 1
    current_email_index: int = 0
    total_emails: int = 0
    total_score: float = 0.0
    average_score: float = 0.0
    completed_emails: List[str] = Field(default_factory=list)
    task_completed: bool = False


@dataclass
class StepResult:
    """Result of a step or reset operation."""
    observation: EmailObservation
    reward: Optional[float] = None
    done: bool = False