---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
sdk_version: "1.0.0"
python_version: "3.11"
app_port: 8000
pinned: false
short_description: Train AI agents on real-world customer support email triage 
tags:
  - openenv
  - reinforcement-learning
  - email-triage
  - customer-support
  - fastapi
  - websocket
  - docker
---
# 📧 Email Triage OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-blue)](https://github.com/openenv)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![WebSocket](https://img.shields.io/badge/WebSocket-Ready-purple)](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)
[![Hugging Face](https://img.shields.io/badge/HF-Space-orange)](https://huggingface.co/spaces)

> **Production-ready, OpenEnv-compliant environment for training AI agents on real-world customer support email triage tasks.**

---

## 📋 Overview

Email Triage OpenEnv simulates a real-world customer support inbox where AI agents classify, prioritize, route, and respond to emails.

### Why Email Triage?

* Real-world utility (30–40% support time spent on triage)
* Measurable outcomes (accuracy, routing, response quality)
* Progressive difficulty (classification → full response)
* Immediate business value

---

## ✨ Features

* 3 Difficulty Levels (Easy → Hard)
* Partial Credit Rewards (0–1 scoring)
* WebSocket + REST API support
* Docker-ready deployment
* OpenEnv compliant
* Baseline agent included
* Reply quality scoring
* Escalation logic
* Penalty system

---

## 🏗️ Architecture

```
AI Agent (RL / Inference)
        │
        ▼
WebSocket / HTTP
        │
        ▼
FastAPI Server (Port 8000)
        │
        ▼
EmailTriageEnvironment
 ├── Test Emails (e001–e005)
 ├── Ground Truth Labels
 └── Scoring Engine
```

---

## 🔄 Communication Flow

```
Agent → Connect (WebSocket)
Agent → Reset
Env   → Email e001
Agent → Step Action
Env   → Reward + Next Email
Agent → Get State
Env   → Episode Summary
```

---

## 📋 Task Breakdown

### Level 1: Easy

* Category (0.5)
* Priority (0.5)

```json
{
  "action": {
    "category": "billing",
    "priority": "high"
  }
}
```

---

### Level 2: Medium

* Category (0.4)
* Priority (0.3)
* Department (0.3)

```json
{
  "action": {
    "category": "billing",
    "priority": "high",
    "department": "finance"
  }
}
```

---

### Level 3: Hard

* Category (0.25)
* Priority (0.20)
* Department (0.20)
* Escalation (0.15)
* Reply (0.20)

```json
{
  "action": {
    "category": "billing",
    "priority": "high",
    "department": "finance",
    "needs_escalation": false,
    "reply_draft": "Thank you for contacting us. We will process your refund within 24 hours."
  }
}
```

---

## 🚀 Quick Start

### Prerequisites

* Python 3.10+
* Docker (optional)
* OpenAI API key

---

### Installation

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/email-triage-openenv
cd email-triage-openenv

pip install -e .
pip install -r requirements.txt
```

---

### Run Locally

```bash
uvicorn email_triage_openenv.server.app:app --reload --host 0.0.0.0 --port 8000
```

---

### Test Script

```python
import asyncio
from email_triage_openenv import EmailTriageEnv
from email_triage_openenv.models import EmailAction

async def test():
    env = EmailTriageEnv(base_url='http://localhost:8000')
    result = await env.reset(task_level=1)
    print(result.observation.subject)

    action = EmailAction(category='billing', priority='high')
    result = await env.step(action)
    print(result.reward)

    await env.close()

asyncio.run(test())
```

---

### Run Baseline

```bash
export OPENAI_API_KEY="your-api-key"
export MODEL_NAME="gpt-3.5-turbo"
python inference.py
```

---

## 🐳 Docker

```bash
docker build -t email-triage-env .
docker run -d -p 8000:8000 email-triage-env
```

---

## 📡 API Reference

### Endpoints

| Endpoint | Method | Description   |
| -------- | ------ | ------------- |
| /        | GET    | Info          |
| /health  | GET    | Health check  |
| /reset   | POST   | Start episode |
| /step    | POST   | Submit action |
| /state   | GET    | Get state     |
| /docs    | GET    | Swagger UI    |

---

### WebSocket

```
ws://localhost:8000/ws
```

---

## 📦 Data Models

### EmailAction

```json
{
  "category": "billing | technical_support | general_inquiry | complaint | feedback",
  "priority": "low | medium | high | urgent",
  "department": "finance | engineering | sales | product | support",
  "reply_draft": "string",
  "needs_escalation": "boolean"
}
```

---

### EmailObservation

```json
{
  "email_id": "e001",
  "subject": "...",
  "body": "...",
  "sender": "...",
  "task_level": 1
}
```

---

### EmailState

```json
{
  "episode_id": "uuid",
  "step_count": 1,
  "total_score": 0.8
}
```

---

## 💰 Reward Function

```
Score =
 category + priority + department + escalation + reply_quality - penalties
```

---

### Reply Quality

* Tone (0.3)
* Actionability (0.3)
* Length (0.2)
* Escalation mention (0.2)

---

### Penalties

* Invalid category → -0.1
* Invalid priority → -0.1

---




### cURL Examples

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/reset \
-H "Content-Type: application/json" \
-d '{"task_level":1}'

curl -X POST http://localhost:8000/step \
-H "Content-Type: application/json" \
-d '{"action":{"category":"billing","priority":"high"}}'
```

---

## 📈 Baseline Performance

| Level  | Score | Success |
| ------ | ----- | ------- |
| Easy   | 0.85  | 85%     |
| Medium | 0.72  | 72%     |
| Hard   | 0.58  | 58%     |

---

## 🚢 Deployment

```bash
openenv push --repo-id YOUR_USERNAME/email-triage-openenv
```

---

## 📁 Project Structure

```
email-triage-openenv/
├── Dockerfile
├── inference.py
├── openenv.yaml
├── requirements.txt
└── email_triage_openenv/
    ├── client.py
    ├── models.py
    ├── graders.py
    └── server/
```

---

## 🔧 Environment Variables

| Variable       | Description |
| -------------- | ----------- |
| OPENAI_API_KEY | API key     |
| MODEL_NAME     | LLM model   |
| TASK_LEVEL     | 1–3         |
| ENV_URL        | Server URL  |

---

## 🐛 Troubleshooting

| Issue               | Fix              |
| ------------------- | ---------------- |
| ModuleNotFoundError | pip install -e . |
| Connection refused  | Start server     |
| WebSocket failed    | Check logs       |

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments

* OpenEnv
* FastAPI
* Hugging Face Spaces

---

## 📊 Quick Commands

```bash
uvicorn email_triage_openenv.server.app:app --reload
curl http://localhost:8000/health
python inference.py
openenv push
```

---


