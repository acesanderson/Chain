from rich.console import Console
import datetime

class PlainProgressHandler:
   def emit_started(self, model, query_preview):
       timestamp = datetime.datetime.now().strftime("%H:%M:%S")
       print(f"[{timestamp}] [{model}] Starting: {query_preview}")
   
   def emit_complete(self, model, query_preview, duration):
       timestamp = datetime.datetime.now().strftime("%H:%M:%S")
       print(f"[{timestamp}] [{model}] Complete: ({duration:.1f}s)")
   
   def emit_failed(self, model, query_preview, error):
       timestamp = datetime.datetime.now().strftime("%H:%M:%S")
       print(f"[{timestamp}] [{model}] Failed: {error}")
   
   def emit_canceled(self, model, query_preview):
       timestamp = datetime.datetime.now().strftime("%H:%M:%S")
       print(f"[{timestamp}] [{model}] Canceled: {query_preview}")

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

