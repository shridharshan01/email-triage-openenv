"""FastAPI server for Email Triage environment."""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import uuid
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Direct import
from email_triage_openenv.server.environment import EmailTriageEnvironment
from email_triage_openenv.models import EmailAction


# ========== Request Models ==========
class ResetRequest(BaseModel):
    task_level: int = 1


class StepRequest(BaseModel):
    action: dict


# ========== FastAPI App ==========
app = FastAPI(
    title="Email Triage Environment",
    version="1.0.0",
    description="OpenEnv-compliant email triage for training AI agents",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active environment
_active_env: Optional[EmailTriageEnvironment] = None


# ========== HTTP Endpoints ==========

@app.get("/")
async def root():
    return {
        "name": "Email Triage Environment",
        "version": "1.0.0",
        "endpoints": {
            "websocket": "/ws",
            "reset": "POST /reset or GET /reset?task_level=1",
            "step": "POST /step",
            "state": "GET /state",
            "health": "GET /health"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/reset")
async def http_reset_post(request: Optional[ResetRequest] = None):
    """POST /reset - Start a new episode."""
    global _active_env

    task_level = 1
    if request and request.task_level:
        task_level = request.task_level

    _active_env = EmailTriageEnvironment()
    observation = await _active_env.reset(task_level=task_level)

    return observation.model_dump()


@app.get("/reset")
async def http_reset_get(task_level: int = 1):
    """GET /reset - Alternative for validators."""
    global _active_env

    _active_env = EmailTriageEnvironment()
    observation = await _active_env.reset(task_level=task_level)

    return observation.model_dump()


@app.post("/step")
async def http_step(request: StepRequest):
    """POST /step - Process an action."""
    global _active_env

    if _active_env is None:
        raise HTTPException(status_code=400, detail="Environment not reset. Call /reset first.")

    action = EmailAction(**request.action)
    observation = await _active_env.step(action)

    return {
        "observation": observation.model_dump(),
        "reward": observation.reward,
        "done": observation.done
    }


@app.get("/state")
async def http_state():
    """GET /state - Get current episode state."""
    global _active_env

    if _active_env is None:
        raise HTTPException(status_code=400, detail="Environment not reset. Call /reset first.")

    state = _active_env.state
    return state.model_dump()


# ========== WebSocket Endpoint ==========

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    env = EmailTriageEnvironment()

    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)

            msg_type = request.get("type")
            msg_id = request.get("id", 0)

            if msg_type == "reset":
                task_level = request.get("task_level", 1)
                observation = await env.reset(task_level=task_level)

                response = {
                    "id": msg_id,
                    "type": "reset",
                    "observation": observation.model_dump(),
                    "done": observation.done,
                    "reward": observation.reward
                }
                await websocket.send_text(json.dumps(response))

            elif msg_type == "step":
                action_data = request.get("action", {})
                action = EmailAction(**action_data)
                observation = await env.step(action)

                response = {
                    "id": msg_id,
                    "type": "step",
                    "observation": observation.model_dump(),
                    "done": observation.done,
                    "reward": observation.reward
                }
                await websocket.send_text(json.dumps(response))

            elif msg_type == "state":
                state = env.state
                response = {
                    "id": msg_id,
                    "type": "state",
                    "state": state.model_dump()
                }
                await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        pass