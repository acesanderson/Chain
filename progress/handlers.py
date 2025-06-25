from datetime import datetime
import time


class RichProgressHandler:
    """Rich-based progress handler with spinners and colors"""

    def __init__(self, console):
        self.console = console
        self.concurrent_mode = False
        self.concurrent_line_printed = False

    # Existing individual operation methods...
    def show_spinner(self, model_name, query_preview):
        """Show Rich spinner with live status"""
        if self.concurrent_mode:
            return  # Suppress individual operations during concurrent mode
        
        self.console.print(
            f"⠋ {model_name} | {query_preview}",
            end="\r",
            highlight=False,
            soft_wrap=True,
        )

    def show_complete(self, model_name, query_preview, duration):
        """Update same line with green checkmark, keeping context"""
        if self.concurrent_mode:
            return  # Suppress individual operations during concurrent mode
            
        self.console.print(
            f"✓ {model_name} | {query_preview} | ({duration:.1f}s)", style="green"
        )

    def show_canceled(self, model_name, query_preview):
        """Update same line with warning, keeping context"""
        if self.concurrent_mode:
            return
            
        self.console.print(
            f"⚠ {model_name} | {query_preview} | Canceled", style="yellow"
        )

    def show_cached(self, model_name, query_preview, duration):
        """Show cache hit with lightning symbol"""
        if self.concurrent_mode:
            return  # Suppress individual operations during concurrent mode

        self.console.print(
            f"⚡ {model_name} | {query_preview} | Cached ({duration:.1f}s)", 
            style="cyan"
        )

    def emit_cached(self, model_name, query_preview, duration):
        """Fallback method for backwards compatibility"""
        self.show_cached(model_name, query_preview, duration)


    def show_failed(self, model_name, query_preview, error):
        """Update same line with error, keeping context"""
        if self.concurrent_mode:
            return
            
        self.console.print(
            f"✗ {model_name} | {query_preview} | Failed: {error}", style="red"
        )

    # New concurrent operation methods
    def handle_concurrent_start(self, total: int):
        """Handle start of concurrent operations"""
        self.concurrent_mode = True
        self.concurrent_line_printed = False
        self.console.print(f"⠋ Running {total} concurrent requests...", end="\r", highlight=False)

    def update_concurrent_progress(self, completed: int, total: int, running: int, failed: int, elapsed: float):
        """Update live concurrent progress"""
        if not self.concurrent_mode:
            return
            
        # Only show progress updates every 0.5 seconds to avoid spam
        current_time = time.time()
        if not hasattr(self, '_last_update') or current_time - self._last_update > 0.5:
            self._last_update = current_time
            
            progress_text = f"⠋ Progress: {completed}/{total} complete | {running} running | {failed} failed | {elapsed:.1f}s elapsed"
            self.console.print(progress_text, end="\r", highlight=False)

    def handle_concurrent_complete(self, successful: int, total: int, duration: float):
        """Handle completion of all concurrent operations"""
        self.concurrent_mode = False
        
        if successful == total:
            self.console.print(f"[green]✓[/green] All requests complete: {successful}/{total} successful in {duration:.1f}s")
        else:
            failed = total - successful
            self.console.print(f"[yellow]✓[/yellow] All requests complete: {successful}/{total} successful, {failed} failed in {duration:.1f}s")

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

    def __init__(self):
        self.concurrent_mode = False

    # Existing individual operation methods...
    def show_spinner(self, model_name, query_preview):
        """Show starting state (plain text - no spinner)"""
        if self.concurrent_mode:
            return  # Suppress individual operations during concurrent mode
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{model_name}] Starting: {query_preview}")

    def show_complete(self, model_name, query_preview, duration):
        """Show completion on new line"""
        if self.concurrent_mode:
            return
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{model_name}] Complete: ({duration:.1f}s)")

    def show_canceled(self, model_name, query_preview):
        """Show cancellation on new line"""
        if self.concurrent_mode:
            return
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{model_name}] Canceled")

    def show_cached(self, model_name, query_preview, duration):
        """Show cache hit in plain text"""
        if self.concurrent_mode:
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{model_name}] Cache hit: {query_preview}")

    def emit_cached(self, model_name, query_preview, duration):
        """Fallback method for backwards compatibility"""
        self.show_cached(model_name, query_preview, duration)


    def show_failed(self, model_name, query_preview, error):
        """Show failure on new line"""
        if self.concurrent_mode:
            return
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{model_name}] Failed: {error}")

    # New concurrent operation methods
    def handle_concurrent_start(self, total: int):
        """Handle start of concurrent operations"""
        self.concurrent_mode = True
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] Starting: {total} concurrent requests")

    def update_concurrent_progress(self, completed: int, total: int, running: int, failed: int, elapsed: float):
        """Plain console doesn't show live updates - too noisy"""
        pass  # Plain console shows only start/end messages

    def handle_concurrent_complete(self, successful: int, total: int, duration: float):
        """Handle completion of all concurrent operations"""
        self.concurrent_mode = False
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if successful == total:
            print(f"[{timestamp}] All requests complete: {successful}/{total} successful in {duration:.1f}s")
        else:
            failed = total - successful
            print(f"[{timestamp}] All requests complete: {successful}/{total} successful, {failed} failed in {duration:.1f}s")

    # Fallback methods for backwards compatibility  
    def emit_started(self, model_name, query_preview):
        self.show_spinner(model_name, query_preview)

    def emit_complete(self, model_name, query_preview, duration):
        self.show_complete(model_name, query_preview, duration)

    def emit_canceled(self, model_name, query_preview):
        self.show_canceled(model_name, query_preview)

    def emit_failed(self, model_name, query_preview, error):
        self.show_failed(model_name, query_preview, error)
