"""OpenEnv-compliant client for Email Triage environment."""

import json
import uuid
from typing import Optional, Dict, Any
import websockets
from .models import EmailAction, EmailObservation, EmailState, StepResult
from .models import EmailAction, EmailObservation, EmailState


class EmailTriageEnv:
    """Async client for Email Triage environment."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self._websocket = None
        self._msg_id = 0
        self._episode_id = None

    def _get_next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    async def _ensure_connection(self):
        """Ensure WebSocket connection is established."""
        if self._websocket is None:
            ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
            if not ws_url.endswith("/ws"):
                ws_url = ws_url.rstrip("/") + "/ws"
            self._websocket = await websockets.connect(ws_url)

    async def reset(self, task_level: int = 1, **kwargs) -> Any:
        """Reset the environment."""
        await self._ensure_connection()

        self._episode_id = str(uuid.uuid4())

        request = {
            "id": self._get_next_id(),
            "type": "reset",
            "episode_id": self._episode_id,
            "task_level": task_level
        }

        await self._websocket.send(json.dumps(request))
        response = json.loads(await self._websocket.recv())

        from .models import StepResult

        obs_data = response.get("observation", {})
        observation = EmailObservation(
            done=response.get("done", False),
            reward=response.get("reward"),
            email_id=obs_data.get("email_id", ""),
            subject=obs_data.get("subject", ""),
            body=obs_data.get("body", ""),
            sender=obs_data.get("sender", ""),
            sender_name=obs_data.get("sender_name", ""),
            timestamp=obs_data.get("timestamp", ""),
            task_level=obs_data.get("task_level", task_level),
            email_index=obs_data.get("email_index", 0),
            total_emails=obs_data.get("total_emails", 0),
            thread_history=obs_data.get("thread_history", []),
            previous_actions=obs_data.get("previous_actions", []),
            feedback=obs_data.get("feedback", ""),
            score_breakdown=obs_data.get("score_breakdown", {})
        )

        return StepResult(observation=observation, reward=response.get("reward"), done=response.get("done", False))

    async def step(self, action: EmailAction, **kwargs) -> Any:
        """Take a step in the environment."""
        await self._ensure_connection()

        request = {
            "id": self._get_next_id(),
            "type": "step",
            "action": action.model_dump(exclude_none=True)
        }

        await self._websocket.send(json.dumps(request))
        response = json.loads(await self._websocket.recv())

        from .models import StepResult

        obs_data = response.get("observation", {})
        observation = EmailObservation(
            done=response.get("done", False),
            reward=response.get("reward"),
            email_id=obs_data.get("email_id", ""),
            subject=obs_data.get("subject", ""),
            body=obs_data.get("body", ""),
            sender=obs_data.get("sender", ""),
            sender_name=obs_data.get("sender_name", ""),
            timestamp=obs_data.get("timestamp", ""),
            task_level=obs_data.get("task_level", 1),
            email_index=obs_data.get("email_index", 0),
            total_emails=obs_data.get("total_emails", 0),
            thread_history=obs_data.get("thread_history", []),
            previous_actions=obs_data.get("previous_actions", []),
            feedback=obs_data.get("feedback", ""),
            score_breakdown=obs_data.get("score_breakdown", {})
        )

        return StepResult(observation=observation, reward=response.get("reward"), done=response.get("done", False))

    async def state(self) -> EmailState:
        """Get current state."""
        await self._ensure_connection()

        request = {
            "id": self._get_next_id(),
            "type": "state"
        }

        await self._websocket.send(json.dumps(request))
        response = json.loads(await self._websocket.recv())

        state_data = response.get("state", {})
        return EmailState(
            episode_id=state_data.get("episode_id"),
            step_count=state_data.get("step_count", 0),
            task_level=state_data.get("task_level", 1),
            current_email_index=state_data.get("current_email_index", 0),
            total_emails=state_data.get("total_emails", 0),
            total_score=state_data.get("total_score", 0.0),
            average_score=state_data.get("average_score", 0.0),
            completed_emails=state_data.get("completed_emails", []),
            task_completed=state_data.get("task_completed", False)
        )

    async def close(self):
        """Close the connection."""
        if self._websocket:
            await self._websocket.close()
            self._websocket = None