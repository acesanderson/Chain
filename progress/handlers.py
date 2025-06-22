from rich.console import Console
import datetime

class PlainProgressHandler:
    """Plain text progress handler"""
    
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
    """Rich console progress handler"""
    
    def __init__(self, console: Console):
        self.console = console

    def emit_started(self, model, query_preview):
        self.console.print(f"[yellow]⠋[/yellow] {model} | {query_preview}")

    def emit_complete(self, model, query_preview, duration):
        duration_str = f"({duration:.1f}s)" if duration else ""
        self.console.print(f"[green]✓[/green] {model} {duration_str}")

    def emit_failed(self, model, query_preview, error):
        self.console.print(f"[red]✗[/red] {model} | {error}")

    def emit_canceled(self, model, query_preview):
        self.console.print(f"[yellow]⚠[/yellow] {model} | Canceled")
