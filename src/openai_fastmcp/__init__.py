"""FastMCP server that exposes OpenAI and Azure OpenAI endpoints."""

from .app import create_server, run
from .config import Settings

__all__ = ["create_server", "run", "Settings"]
