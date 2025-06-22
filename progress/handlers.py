from rich.console import Console

class PlainProgressHandler:
    def handle_event(self, event):
        timestamp = event.timestamp.strftime("%H:%M:%S")
        
        if event.event_type == "started":
            print(f"[{timestamp}] Starting: {event.model} | {event.query_preview}")
        elif event.event_type == "complete":
            duration = f"({event.duration:.1f}s)" if event.duration else ""
            print(f"[{timestamp}] Complete: {event.model} {duration}")
        elif event.event_type == "failed":
            print(f"[{timestamp}] Failed: {event.model} | {event.error}")

class RichProgressHandler:
    def __init__(self, console: Console):
        self.console = console
    
    def handle_event(self, event):
        if event.event_type == "started":
            self.console.print(f"[yellow]⠋[/yellow] {event.model} | {event.query_preview}")
        elif event.event_type == "complete":
            duration = f"({event.duration:.1f}s)" if event.duration else ""
            self.console.print(f"[green]✓[/green] {event.model} {duration}")
        elif event.event_type == "failed":
            self.console.print(f"[red]✗[/red] {event.model} | {event.error}")

