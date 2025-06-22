from rich.console import Console
from datetime import datetime

class PlainProgressHandler:
    """Simple progress handler for environments without Rich"""
    
    def show_spinner(self, model_name, query_preview):
        """Show spinner state (plain text version)"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{model_name}] Starting: {query_preview}", end="", flush=True)
    
    def show_complete(self, model_name, query_preview, duration):
        """Update same line with completion"""
        print(f"\r[{datetime.now().strftime('%H:%M:%S')}] [{model_name}] Complete: ({duration:.1f}s)")
    
    def show_canceled(self, model_name, query_preview):
        """Update same line with cancellation"""
        print(f"\r[{datetime.now().strftime('%H:%M:%S')}] [{model_name}] Canceled")
    
    def show_failed(self, model_name, query_preview, error):
        """Update same line with failure"""
        print(f"\r[{datetime.now().strftime('%H:%M:%S')}] [{model_name}] Failed: {error}")

    # Fallback methods for backwards compatibility
    def emit_started(self, model_name, query_preview):
        self.show_spinner(model_name, query_preview)
    
    def emit_complete(self, model_name, query_preview, duration):
        self.show_complete(model_name, query_preview, duration)
    
    def emit_canceled(self, model_name, query_preview):
        self.show_canceled(model_name, query_preview)
    
    def emit_failed(self, model_name, query_preview, error):
        self.show_failed(model_name, query_preview, error)


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
