"""FastAPI dependencies: access to the prepared AppState singleton."""

from __future__ import annotations

from app.services.app_state import AppState, get_app_state


def get_state() -> AppState:
    """FastAPI dependency returning the prepared AppState."""
    return get_app_state()