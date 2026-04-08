#!/usr/bin/env python3
"""Server entry point for OpenEnv validator."""

import sys
import os
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from email_triage_openenv.server.app import app
except ImportError:
    from app import app


def main():
    """Main entry point for the server."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()