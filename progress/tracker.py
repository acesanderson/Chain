from datetime import datetime
from pydantic import BaseModel
from typing import Protocol

class ProgressEvent(BaseModel):
    pass

class SyncEvent(ProgressEvent):
    event_type: str  # "started", "complete", "failed", "canceled"
    timestamp: datetime
    model: str
    query_preview: str  # First ~30 chars of the query
    duration: float | None = None  # Duration in seconds
    error: str | None = None  # Error message if any

class AsyncEvent(ProgressEvent):
    request_id: int
    event_type: str  # "started", "complete", "failed", "canceled"
    timestamp: datetime
    model: str
    query_preview: str  # First ~30 chars of the query
    duration: float | None = None  # Duration in seconds
    error: str | None = None  # Error message if any

class ProgressHandler(Protocol):
    def handle_event(self, event: ProgressEvent) -> None:
        """Handle a progress event."""
        ...

class ProgressTracker:
    def __init__(self, handler: ProgressHandler):
        self.handler = handler

    def emit_event(self, event: ProgressEvent):
        self.handler.handle_event(event)

