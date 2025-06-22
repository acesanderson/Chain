from Chain.progress.tracker import ProgressTracker, SyncEvent, AsyncEvent, ProgressEvent
from Chain.progress.handlers import PlainProgressHandler, RichProgressHandler
from rich.console import Console
from datetime import datetime

# Test event
event = SyncEvent(
    event_type="started",
    timestamp=datetime.now(),
    model="gpt-4o",
    query_preview="What is the capital of France?"
)

# Test plain handler
plain_handler = PlainProgressHandler()
tracker = ProgressTracker(plain_handler)
tracker.emit_event(event)

# Test rich handler
rich_handler = RichProgressHandler(Console())
tracker = ProgressTracker(rich_handler)
tracker.emit_event(event)
