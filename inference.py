#!/usr/bin/env python3
"""
Baseline inference script for Email Triage environment.
"""

import asyncio
import os
import json
import sys
import traceback
from typing import List, Dict, Any, Optional
from openai import OpenAI

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ========== CONFIGURATION ==========
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
TASK_LEVEL = int(os.getenv("TASK_LEVEL", "1"))

# Try multiple possible environment URLs
ENV_URL = os.getenv("ENV_URL")
if not ENV_URL:
    # Try local first, then HF Space
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex(('127.0.0.1', 8000))
    if result == 0:
        ENV_URL = "http://localhost:8000"
    else:
        ENV_URL = "https://shridharshan01-email-triage-openenv.hf.space"
    sock.close()

MAX_STEPS = 100
TEMPERATURE = 0.3

# ========== SYSTEM PROMPTS ==========
SYSTEM_PROMPTS = {
    1: """You are an expert email triage assistant for a customer support team.

Your task: Classify each email's CATEGORY and PRIORITY.

Categories: billing, technical_support, general_inquiry, complaint, feedback
Priorities: urgent, high, medium, low

Respond with valid JSON only:
{"category": "...", "priority": "..."}""",

    2: """You are an expert email triage assistant for a customer support team.

Your task: Classify CATEGORY, PRIORITY, and DEPARTMENT.

Respond with valid JSON only:
{"category": "...", "priority": "...", "department": "..."}""",

    3: """You are an expert email triage assistant for a customer support team.

Your task: Classify CATEGORY, PRIORITY, DEPARTMENT, escalation, and reply.

Respond with valid JSON only:
{
    "category": "...",
    "priority": "...",
    "department": "...",
    "needs_escalation": true/false,
    "reply_draft": "..."
}"""
}


# ========== HELPER FUNCTIONS ==========
def build_user_prompt(observation, task_level: int) -> str:
    """Build user prompt from observation."""
    prompt = f"""
Email ID: {observation.email_id}
From: {observation.sender_name} <{observation.sender}>
Subject: {observation.subject}
Body: {observation.body[:500] if observation.body else 'N/A'}
Timestamp: {observation.timestamp}
"""
    prompt += "\n\nClassify this email:"
    return prompt


def parse_model_response(response_text: str, task_level: int):
    """Parse model response into EmailAction."""
    try:
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        data = json.loads(response_text)

        # Import here to avoid circular imports
        from email_triage_openenv.models import EmailAction

        return EmailAction(
            category=data.get("category", "general_inquiry"),
            priority=data.get("priority", "medium"),
            department=data.get("department", "") if task_level >= 2 else "",
            reply_draft=data.get("reply_draft", "") if task_level >= 3 else "",
            needs_escalation=data.get("needs_escalation", False) if task_level >= 3 else False,
            is_duplicate=data.get("is_duplicate", False)
        )
    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse error: {e}", flush=True)
        print(f"[DEBUG] Raw response: {response_text[:200]}", flush=True)
        from email_triage_openenv.models import EmailAction
        return EmailAction(category="general_inquiry", priority="medium")


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ========== MAIN INFERENCE LOOP ==========
async def run_baseline() -> Dict[str, Any]:
    results = {
        "task_level": TASK_LEVEL,
        "total_emails": 0,
        "total_reward": 0.0,
        "rewards": [],
        "actions": [],
        "success": False
    }

    log_start(task=f"level_{TASK_LEVEL}", env="email_triage", model=MODEL_NAME)

    # Validate API key
    if not API_KEY:
        error_msg = "OPENAI_API_KEY or HF_TOKEN environment variable not set"
        print(f"[ERROR] {error_msg}", flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return {"error": error_msg}

    # Initialize OpenAI client
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        print(f"[DEBUG] OpenAI client initialized with model: {MODEL_NAME}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to initialize OpenAI client: {e}", flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return {"error": str(e)}

    # Initialize environment
    try:
        print(f"[DEBUG] Connecting to environment at: {ENV_URL}", flush=True)
        from email_triage_openenv import EmailTriageEnv
        env = EmailTriageEnv(base_url=ENV_URL)
        print(f"[DEBUG] Environment client created successfully", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to connect to environment at {ENV_URL}: {e}", flush=True)
        print(f"[ERROR] Traceback: {traceback.format_exc()}", flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return {"error": str(e)}

    try:
        # Reset environment
        print(f"[DEBUG] Resetting environment for task level {TASK_LEVEL}...", flush=True)
        result = await env.reset(task_level=TASK_LEVEL)
        obs = result.observation
        step = 0
        print(f"[DEBUG] Reset successful. First email: {obs.email_id}", flush=True)

        while not result.done and step < MAX_STEPS:
            user_prompt = build_user_prompt(obs, TASK_LEVEL)

            try:
                # Get model prediction
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPTS[TASK_LEVEL]},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=300,
                )
                action_text = response.choices[0].message.content or "{}"
                action = parse_model_response(action_text, TASK_LEVEL)

                # Take step
                result = await env.step(action)
                obs = result.observation
                step += 1

                reward = result.reward or 0.0
                results["total_reward"] += reward
                results["rewards"].append(reward)
                results["total_emails"] = step

                action_str = json.dumps(action.model_dump(exclude_none=True))
                log_step(step=step, action=action_str, reward=reward, done=result.done, error=None)
                print(f"[DEBUG] Step {step}: {obs.email_id} - Reward: {reward:.2f} - Done: {result.done}", flush=True)

            except Exception as e:
                print(f"[DEBUG] Model or step error: {e}", flush=True)
                log_step(step=step + 1, action="{}", reward=0.0, done=False, error=str(e))
                break

        # Calculate final score
        if results["total_emails"] > 0:
            avg_score = results["total_reward"] / results["total_emails"]
        else:
            avg_score = 0.0

        results["success"] = avg_score >= 0.5

        print(f"\n[DEBUG] ========== EPISODE COMPLETE ==========", flush=True)
        print(f"[DEBUG] Emails Processed: {results['total_emails']}", flush=True)
        print(f"[DEBUG] Total Reward: {results['total_reward']:.2f}", flush=True)
        print(f"[DEBUG] Average Score: {avg_score:.3f}", flush=True)
        print(f"[DEBUG] Success: {results['success']}", flush=True)

        log_end(success=results["success"], steps=results["total_emails"], score=avg_score, rewards=results["rewards"])
        return results

    except Exception as e:
        print(f"[DEBUG] Fatal error: {e}", flush=True)
        traceback.print_exc()
        log_end(success=False, steps=results["total_emails"], score=0.0, rewards=results["rewards"])
        return {"error": str(e)}

    finally:
        await env.close()
        print(f"[DEBUG] Environment closed", flush=True)


# ========== ENTRY POINT ==========
if __name__ == "__main__":
    print(f"[DEBUG] Starting Email Triage Baseline Inference", flush=True)
    print(f"[DEBUG] Model: {MODEL_NAME}", flush=True)
    print(f"[DEBUG] Task Level: {TASK_LEVEL}", flush=True)
    print(f"[DEBUG] Environment URL: {ENV_URL}", flush=True)
    print(f"[DEBUG] API Base URL: {API_BASE_URL}", flush=True)
    print(f"", flush=True)

    try:
        results = asyncio.run(run_baseline())

        # Exit with appropriate code
        if results.get("error"):
            print(f"[ERROR] {results['error']}", flush=True)
            sys.exit(1)
        elif results.get("success"):
            print(f"[INFO] Inference completed successfully", flush=True)
            sys.exit(0)
        else:
            print(f"[INFO] Inference completed but success threshold not met", flush=True)
            sys.exit(0)  # Exit 0 to allow validation to see the output
    except Exception as e:
        print(f"[ERROR] Unhandled exception: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)