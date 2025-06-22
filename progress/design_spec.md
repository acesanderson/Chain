# Chain Progress Display System - Design Specification

## Overview

Add optional progress display functionality to Chain and AsyncChain that provides real-time feedback during LLM operations. The system prioritizes simplicity, maintainability, and backward compatibility.

## Project Structure

```
Chain/
├── progress/
│   ├── __init__.py          # Exports: ProgressTracker, RichProgressHandler, PlainProgressHandler
│   ├── tracker.py           # ~50 lines - Core event tracking logic
│   ├── handlers.py          # ~100 lines - Display implementations  
│   └── wrappers.py          # ~50 lines - Sync/async function wrapping
```

## Display Behavior

### Rich Console Mode (when Console object available)

**Sync Operations:**

```
[⠋] gemini-2.5-flash | completion | "You are a helpful IT admin..." (15.2s)
```

Updates to:

```
[✓] gemini-2.5-flash | completion | "You are a helpful IT admin..." (23.4s)
```

**Async Operations:**

```
Progress: 3/8 complete | 2 running | 0 failed | 00:45 elapsed

[⠋] cogito:14b | completion | "Make a new frog..." (03.2s)
[✓] cogito:14b | completion | "Make a new frog..." (12.1s) 
[⠋] cogito:14b | completion | "Make a new frog..." (01.8s)
[✓] cogito:14b | completion | "Make a new frog..." (08.5s)
[⠋] cogito:14b | completion | "Make a new frog..." (02.3s)
```

### Plain Console Mode (when no Console available, verbose=True)

**Sync Operations:**

```
Starting: gemini-2.5-flash | "You are a helpful IT admin..."
Complete: gemini-2.5-flash | (23.4s)
```

**Async Operations:**

```
Starting batch: 8 requests
Complete: cogito:14b | (12.1s)
Complete: cogito:14b | (08.5s)
Batch complete: 8/8 successful in 45.2s
```

## Architecture

### Three-Layer Design

1. **Event Emission**: Function wrappers emit standardized events
2. **Progress Tracking**: Central coordinator manages state and routing
3. **Display Handlers**: Render events to terminal (Rich vs Plain)

### Event Schema

```python
# Sync events (no request_id needed)
{
    "event_type": "started" | "complete" | "failed" | "canceled",
    "timestamp": datetime,
    "model": str,
    "query_preview": str,  # First ~30 chars
    "duration": float | None,
    "error": str | None
}

# Async events (includes request_id for batch tracking)
{
    "request_id": int,
    "event_type": "started" | "complete" | "failed" | "canceled" | "batch_start" | "batch_complete",
    "timestamp": datetime,
    "model": str,
    "query_preview": str,
    "duration": float | None,
    "batch_total": int | None,
    "error": str | None
}
```

## Console Singleton Pattern

### Class-Level Console Setup

```python
class Chain:
    _console: Console | None = None
    
    def __init__(self, model, prompt=None, parser=None):
        # ... existing code ...
        self.console = None  # Instance-level console

class AsyncChain(Chain):
    _console: Console | None = None  # Inherits from Chain but can override
```

### Console Resolution Hierarchy

1. **Instance-level**: `chain.console = Console()` (highest priority)
2. **Class-level**: `Chain._console = Console()` (fallback)
3. **None**: Plain text display (default)

### Usage Patterns

```python
# Set globally for all Chain instances
Chain._console = Console()

# Set globally for all AsyncChain instances  
AsyncChain._console = Console()

# Override for specific instance
chain = Chain(model=model)
chain.console = None  # This instance uses plain text
```

## Integration Points

### Chain/AsyncChain Modifications

```python
def run(self, input_variables=None, messages=None, verbose=True, **kwargs):
    # ... existing prompt preparation ...
    
    effective_console = self.console or self.__class__._console
    
    if verbose and effective_console:
        # Use Rich progress display
        handler = RichProgressHandler(effective_console)
        tracker = ProgressTracker(handler)
        wrapped_query = wrap_sync_query(self.model.query, tracker)
        result = wrapped_query(prompt, verbose=False, parser=self.parser)
    elif verbose:
        # Use plain text display  
        handler = PlainProgressHandler()
        tracker = ProgressTracker(handler)
        wrapped_query = wrap_sync_query(self.model.query, tracker)
        result = wrapped_query(prompt, verbose=False, parser=self.parser)
    else:
        # No progress display
        result = self.model.query(prompt, verbose=verbose, parser=self.parser)
        
    # ... rest of method unchanged ...
```

### Function Wrapping Strategy

- **Sync**: Wrap `model.query()` call within `Chain.run()`
- **Async**: Wrap each coroutine before `asyncio.gather()` in `AsyncChain._run_*()`
- **No modification** to Model classes - they remain progress-unaware

## Implementation Components

### ProgressTracker (tracker.py)

- Manages event routing to display handlers
- Maintains minimal state for batch operations
- Simple event emission interface
- **Key methods**: `emit_event()`, `start_batch()`, `end_batch()`

### Display Handlers (handlers.py)

- **RichProgressHandler**: Spinners, colors, live updates using Rich library
- **PlainProgressHandler**: Simple print statements with timestamps
- **Common interface**: `handle_event(event_dict)`

### Function Wrappers (wrappers.py)

- **wrap_sync_query()**: Decorator for single operations
- **wrap_async_coroutine()**: Wrapper for batch operations
- Handle timing, error catching, and event emission
- Preserve original function signatures and return values

## Future Extensibility

### Query Type Classification

To add query type display ("completion", "reasoning", "function call"):

- Add `query_type` field to event schema
- Implement classification logic in wrappers based on:
    - Parser presence (function call)
    - Model type (reasoning models)
    - Message content analysis
- Pass as parameter to Chain/AsyncChain if custom classification needed

### SSE Integration (Future)

The event schema is JSON-serializable and the ProgressTracker is display-agnostic, making SSE extension straightforward:

- Create `SSEProgressHandler` that sends events over HTTP
- No changes needed to core tracking or wrapper logic
- Chain server can emit same events to web clients

## Error Handling

- **Scope**: Progress system only handles display of failures, not error recovery
- **KeyboardInterrupt**: Shows "canceled" status and re-raises cleanly
- **Other exceptions**: Shows "failed" status and re-raises
- **No retry logic**: Handled separately from progress concerns

## Backward Compatibility

- **Zero breaking changes**: All existing Chain/AsyncChain method calls work unchanged
- **Console singleton**: New class attribute, existing code unaffected
- **Opt-in behavior**: Progress only activates when verbose=True + Console available
- **Performance**: No overhead when progress disabled

## Implementation Priority

1. **Start with sync** (Chain.run) - simpler execution model
2. **Extend to async** (AsyncChain.run) - reuse event schema and handlers
3. **Test extensively** with existing codebases to ensure compatibility

This design prioritizes simplicity while creating a foundation that can extend to remote progress monitoring and advanced display features as needed.
