from datetime import datetime


class RichProgressHandler:
    """Rich-based progress handler with spinners and colors"""

    def __init__(self, console):
        self.console = console

    def show_spinner(self, model_name, query_preview):
        """Show Rich spinner with live status"""
        self.console.print(
            f"⠋ {model_name} | {query_preview}",
            end="\r",
            highlight=False,
            soft_wrap=True,
        )

    def show_complete(self, model_name, query_preview, duration):
        """Update same line with green checkmark, keeping context"""
        self.console.print(
            f"✓ {model_name} | {query_preview} | ({duration:.1f}s)", style="green"
        )

    def show_canceled(self, model_name, query_preview):
        """Update same line with warning, keeping context"""
        self.console.print(
            f"⚠ {model_name} | {query_preview} | Canceled", style="yellow"
        )

    def show_failed(self, model_name, query_preview, error):
        """Update same line with error, keeping context"""
        self.console.print(
            f"✗ {model_name} | {query_preview} | Failed: {error}", style="red"
        )

    # Fallback methods for backwards compatibility
    def emit_started(self, model_name, query_preview):
        self.show_spinner(model_name, query_preview)

    def emit_complete(self, model_name, query_preview, duration):
        self.show_complete(model_name, query_preview, duration)

    def emit_canceled(self, model_name, query_preview):
        self.show_canceled(model_name, query_preview)

    def emit_failed(self, model_name, query_preview, error):
        self.show_failed(model_name, query_preview, error)


class PlainProgressHandler:
    """Simple progress handler for environments without Rich"""

    def show_spinner(self, model_name, query_preview):
        """Show starting state (plain text - no spinner)"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{model_name}] Starting: {query_preview}")

    def show_complete(self, model_name, query_preview, duration):
        """Show completion on new line"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{model_name}] Complete: ({duration:.1f}s)")

    def show_canceled(self, model_name, query_preview):
        """Show cancellation on new line"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{model_name}] Canceled")

    def show_failed(self, model_name, query_preview, error):
        """Show failure on new line"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{model_name}] Failed: {error}")

    # Fallback methods for backwards compatibility
    def emit_started(self, model_name, query_preview):
        self.show_spinner(model_name, query_preview)

    def emit_complete(self, model_name, query_preview, duration):
        self.show_complete(model_name, query_preview, duration)

    def emit_canceled(self, model_name, query_preview):
        self.show_canceled(model_name, query_preview)

    def emit_failed(self, model_name, query_preview, error):
        self.show_failed(model_name, query_preview, error)
