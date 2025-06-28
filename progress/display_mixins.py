"""
Display mixins for Chain verbosity system.

These mixins provide rich and plain text formatting capabilities to data objects
like Params, Response, and ChainError. Each mixin handles the specific display
logic for its object type across different verbosity levels.
"""

from Chain.progress.verbosity import Verbosity
from typing import TYPE_CHECKING, Any
import json

# TYPE_CHECKING imports to avoid circular dependencies
if TYPE_CHECKING:
    from rich.console import RenderableType
    from rich.panel import Panel
    from rich.syntax import Syntax


class RichDisplayMixin:
    """Base mixin for Rich console display functionality."""
    
    def to_rich(self, verbosity: Verbosity) -> "RenderableType":
        """
        Convert object to Rich renderable based on verbosity level.
        
        Args:
            verbosity: The verbosity level for display
            
        Returns:
            Rich renderable object (Panel, Syntax, Text, etc.)
        """
        if verbosity == Verbosity.SILENT:
            return ""
        elif verbosity == Verbosity.PROGRESS:
            # PROGRESS level handled by existing progress system
            return ""
        else:
            # Delegate to specific implementation
            return self._to_rich_impl(verbosity)
    
    def _to_rich_impl(self, verbosity: Verbosity) -> "RenderableType":
        """Override in subclasses for specific Rich formatting logic."""
        raise NotImplementedError("Subclasses must implement _to_rich_impl")


class PlainDisplayMixin:
    """Base mixin for plain text display functionality."""
    
    def to_plain(self, verbosity: Verbosity) -> str:
        """
        Convert object to plain text based on verbosity level.
        
        Args:
            verbosity: The verbosity level for display
            
        Returns:
            Plain text string representation
        """
        if verbosity == Verbosity.SILENT:
            return ""
        elif verbosity == Verbosity.PROGRESS:
            # PROGRESS level handled by existing progress system
            return ""
        else:
            # Delegate to specific implementation
            return self._to_plain_impl(verbosity)
    
    def _to_plain_impl(self, verbosity: Verbosity) -> str:
        """Override in subclasses for specific plain text formatting logic."""
        raise NotImplementedError("Subclasses must implement _to_plain_impl")


class RichDisplayParamsMixin(RichDisplayMixin):
    """Rich display mixin for Params objects."""
    
    def _to_rich_impl(self, verbosity: Verbosity) -> "RenderableType":
        """Format Params object for Rich console display."""
        from rich.panel import Panel
        from rich.syntax import Syntax
        from rich.text import Text
        from rich.table import Table
        
        if verbosity == Verbosity.SUMMARY:
            return self._format_params_summary_rich()
        elif verbosity == Verbosity.DETAILED:
            return self._format_params_detailed_rich()
        elif verbosity == Verbosity.COMPLETE:
            return self._format_params_complete_rich()
        elif verbosity == Verbosity.DEBUG:
            return self._format_params_debug_rich()
        else:
            return Text("")
    
    def _format_params_summary_rich(self) -> "Panel":
        """Format basic request info for SUMMARY level."""
        from rich.panel import Panel
        from rich.text import Text
        from datetime import datetime
        
        # Build content text
        content = Text()
        
        # Show user message(s)
        if hasattr(self, 'messages') and self.messages:
            for msg in self.messages:
                if hasattr(msg, 'role') and msg.role == 'user':
                    content.append(f"user: {str(msg.content)}\n", style="yellow")
        elif hasattr(self, 'query_input') and self.query_input:
            content.append(f"user: {str(self.query_input)}\n", style="yellow")
        
        # Show parameters
        params_line = []
        if hasattr(self, 'temperature') and self.temperature is not None:
            params_line.append(f"Temperature: {self.temperature}")
        if hasattr(self, 'parser') and self.parser:
            parser_name = getattr(self.parser, 'pydantic_model', {})
            if hasattr(parser_name, '__name__'):
                params_line.append(f"Parser: {parser_name.__name__}")
        
        if params_line:
            content.append(" • ".join(params_line), style="dim")
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        return Panel(content, title=f"► REQUEST {getattr(self, 'model', 'unknown')}", 
                    title_align="left", subtitle=f"[dim]{timestamp}[/dim]", subtitle_align="right")
    
    def _format_params_detailed_rich(self) -> "Panel":
        """Format truncated messages for DETAILED level."""
        # TODO: Implement - show messages table with truncation (~100 chars)
        from rich.panel import Panel
        from rich.text import Text
        content = Text("REQUEST DETAILED - TODO: Implement")
        return Panel(content, title="► REQUEST (Detailed)")
    
    def _format_params_complete_rich(self) -> "Panel":
        """Format complete messages for COMPLETE level."""
        # TODO: Implement - show full messages with word wrapping
        from rich.panel import Panel
        from rich.text import Text
        content = Text("REQUEST COMPLETE - TODO: Implement")
        return Panel(content, title="► REQUEST (Complete)")
    
    def _format_params_debug_rich(self) -> "Panel":
        """Format full JSON debug for DEBUG level."""
        # TODO: Implement - show syntax highlighted JSON with line numbers
        from rich.panel import Panel
        from rich.syntax import Syntax
        json_content = json.dumps({"debug": "TODO: Implement"}, indent=2)
        syntax = Syntax(json_content, "json", line_numbers=True)
        return Panel(syntax, title="🐛 FULL DEBUG REQUEST")


class PlainDisplayParamsMixin(PlainDisplayMixin):
    """Plain text display mixin for Params objects."""
    
    def _to_plain_impl(self, verbosity: Verbosity) -> str:
        """Format Params object for plain text display."""
        if verbosity == Verbosity.SUMMARY:
            return self._format_params_summary_plain()
        elif verbosity == Verbosity.DETAILED:
            return self._format_params_detailed_plain()
        elif verbosity == Verbosity.COMPLETE:
            return self._format_params_complete_plain()
        elif verbosity == Verbosity.DEBUG:
            return self._format_params_debug_plain()
        else:
            return ""
    
    def _format_params_summary_plain(self) -> str:
        """Format basic request info for SUMMARY level."""
        from datetime import datetime
        
        lines = []
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Header
        lines.append(f"[{timestamp}] REQUEST {getattr(self, 'model', 'unknown')}")
        
        # Show user message
        if hasattr(self, 'messages') and self.messages:
            for msg in self.messages:
                if hasattr(msg, 'role') and msg.role == 'user':
                    lines.append(f"user: {str(msg.content)}")
        elif hasattr(self, 'query_input') and self.query_input:
            lines.append(f"user: {str(self.query_input)}")
        
        # Show parameters
        params_line = []
        if hasattr(self, 'temperature') and self.temperature is not None:
            params_line.append(f"Temperature: {self.temperature}")
        if hasattr(self, 'parser') and self.parser:
            parser_name = getattr(self.parser, 'pydantic_model', {})
            if hasattr(parser_name, '__name__'):
                params_line.append(f"Parser: {parser_name.__name__}")
        
        if params_line:
            lines.append(" • ".join(params_line))
        
        return "\n".join(lines)
    
    def _format_params_detailed_plain(self) -> str:
        """Format truncated messages for DETAILED level."""
        # TODO: Implement - show messages with truncation
        return "REQUEST DETAILED - TODO: Implement"
    
    def _format_params_complete_plain(self) -> str:
        """Format complete messages for COMPLETE level."""
        # TODO: Implement - show full messages
        return "REQUEST COMPLETE - TODO: Implement"
    
    def _format_params_debug_plain(self) -> str:
        """Format full JSON debug for DEBUG level."""
        # TODO: Implement - show formatted JSON
        return "DEBUG REQUEST - TODO: Implement"


class RichDisplayResponseMixin(RichDisplayMixin):
    """Rich display mixin for Response objects."""
    
    def _to_rich_impl(self, verbosity: Verbosity) -> "RenderableType":
        """Format Response object for Rich console display."""
        from rich.panel import Panel
        from rich.syntax import Syntax
        from rich.text import Text
        
        if verbosity == Verbosity.SUMMARY:
            return self._format_response_summary_rich()
        elif verbosity == Verbosity.DETAILED:
            return self._format_response_detailed_rich()
        elif verbosity == Verbosity.COMPLETE:
            return self._format_response_complete_rich()
        elif verbosity == Verbosity.DEBUG:
            return self._format_response_debug_rich()
        else:
            return Text("")
    
    def _format_response_summary_rich(self) -> "Panel":
        """Format basic response info for SUMMARY level."""
        from rich.panel import Panel
        from rich.text import Text
        from datetime import datetime
        
        content = Text()
        duration = getattr(self, 'duration', 0)
        
        # Show response content (truncated)
        response_content = str(getattr(self, 'content', 'No content'))
        if len(response_content) > 100:
            response_content = response_content[:100] + "..."
        
        content.append(response_content, style="blue")
        
        # Show metadata if available
        if hasattr(self, 'model'):
            content.append(f"\nModel: {self.model}", style="dim")
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        return Panel(content, title=f"✓ RESPONSE {duration:.1f}s", 
                    title_align="left", subtitle=f"[dim]{timestamp}[/dim]", subtitle_align="right")
    
    def _format_response_detailed_rich(self) -> "Panel":
        """Format truncated response for DETAILED level."""
        # TODO: Implement - show content with truncation
        from rich.panel import Panel
        from rich.text import Text
        content = Text("RESPONSE DETAILED - TODO: Implement")
        return Panel(content, title="✓ RESPONSE (Detailed)")
    
    def _format_response_complete_rich(self) -> "Panel":
        """Format complete response for COMPLETE level."""
        # TODO: Implement - show full response content
        from rich.panel import Panel
        from rich.text import Text
        content = Text("RESPONSE COMPLETE - TODO: Implement")
        return Panel(content, title="✓ RESPONSE (Complete)")
    
    def _format_response_debug_rich(self) -> "Panel":
        """Format full JSON debug for DEBUG level."""
        # TODO: Implement - show syntax highlighted JSON with metadata
        from rich.panel import Panel
        from rich.syntax import Syntax
        duration = getattr(self, 'duration', 0)
        json_content = json.dumps({"debug": "TODO: Implement"}, indent=2)
        syntax = Syntax(json_content, "json", line_numbers=True)
        return Panel(syntax, title=f"🐛 FULL DEBUG RESPONSE {duration:.1f}s")


class PlainDisplayResponseMixin(PlainDisplayMixin):
    """Plain text display mixin for Response objects."""
    
    def _to_plain_impl(self, verbosity: Verbosity) -> str:
        """Format Response object for plain text display."""
        if verbosity == Verbosity.SUMMARY:
            return self._format_response_summary_plain()
        elif verbosity == Verbosity.DETAILED:
            return self._format_response_detailed_plain()
        elif verbosity == Verbosity.COMPLETE:
            return self._format_response_complete_plain()
        elif verbosity == Verbosity.DEBUG:
            return self._format_response_debug_plain()
        else:
            return ""
    
    def _format_response_summary_plain(self) -> str:
        """Format basic response info for SUMMARY level."""
        from datetime import datetime
        
        lines = []
        timestamp = datetime.now().strftime("%H:%M:%S")
        duration = getattr(self, 'duration', 0)
        
        # Header
        lines.append(f"[{timestamp}] RESPONSE {duration:.1f}s")
        
        # Show response content (truncated)
        response_content = str(getattr(self, 'content', 'No content'))
        if len(response_content) > 100:
            response_content = response_content[:100] + "..."
        lines.append(response_content)
        
        return "\n".join(lines)
    
    def _format_response_detailed_plain(self) -> str:
        """Format truncated response for DETAILED level."""
        # TODO: Implement - show content with truncation
        return "RESPONSE DETAILED - TODO: Implement"
    
    def _format_response_complete_plain(self) -> str:
        """Format complete response for COMPLETE level."""
        # TODO: Implement - show full response
        return "RESPONSE COMPLETE - TODO: Implement"
    
    def _format_response_debug_plain(self) -> str:
        """Format full JSON debug for DEBUG level."""
        # TODO: Implement - show formatted JSON
        return "DEBUG RESPONSE - TODO: Implement"


class RichDisplayChainErrorMixin(RichDisplayMixin):
    """Rich display mixin for ChainError objects."""
    
    def _to_rich_impl(self, verbosity: Verbosity) -> "RenderableType":
        """Format ChainError object for Rich console display."""
        from rich.panel import Panel
        from rich.syntax import Syntax
        from rich.text import Text
        
        if verbosity == Verbosity.SUMMARY:
            return self._format_error_summary_rich()
        elif verbosity == Verbosity.DETAILED:
            return self._format_error_detailed_rich()
        elif verbosity == Verbosity.COMPLETE:
            return self._format_error_complete_rich()
        elif verbosity == Verbosity.DEBUG:
            return self._format_error_debug_rich()
        else:
            return Text("")
    
    def _format_error_summary_rich(self) -> "Panel":
        """Format basic error info for SUMMARY level."""
        from rich.panel import Panel
        from rich.text import Text
        from datetime import datetime
        
        content = Text()
        
        # Show error info
        if hasattr(self, 'info'):
            content.append(f"Error: {self.info.code}\n", style="bold red")
            content.append(f"{self.info.message}\n", style="red")
            content.append(f"Category: {self.info.category}", style="dim red")
        else:
            content.append("Unknown error", style="red")
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        return Panel(content, title="✗ ERROR", title_align="left", 
                    subtitle=f"[dim]{timestamp}[/dim]", subtitle_align="right", border_style="red")
    
    def _format_error_detailed_rich(self) -> "Panel":
        """Format detailed error for DETAILED level."""
        # TODO: Implement - show error with context
        from rich.panel import Panel
        from rich.text import Text
        content = Text("ERROR DETAILED - TODO: Implement")
        return Panel(content, title="✗ ERROR (Detailed)", border_style="red")
    
    def _format_error_complete_rich(self) -> "Panel":
        """Format complete error for COMPLETE level."""
        # TODO: Implement - show full error with stack trace
        from rich.panel import Panel
        from rich.text import Text
        content = Text("ERROR COMPLETE - TODO: Implement")
        return Panel(content, title="✗ ERROR (Complete)", border_style="red")
    
    def _format_error_debug_rich(self) -> "Panel":
        """Format full JSON debug for DEBUG level."""
        # TODO: Implement - show full error object as JSON
        from rich.panel import Panel
        from rich.syntax import Syntax
        json_content = json.dumps({"error": "TODO: Implement"}, indent=2)
        syntax = Syntax(json_content, "json", line_numbers=True)
        return Panel(syntax, title="🐛 FULL DEBUG ERROR", border_style="red")


class PlainDisplayChainErrorMixin(PlainDisplayMixin):
    """Plain text display mixin for ChainError objects."""
    
    def _to_plain_impl(self, verbosity: Verbosity) -> str:
        """Format ChainError object for plain text display."""
        if verbosity == Verbosity.SUMMARY:
            return self._format_error_summary_plain()
        elif verbosity == Verbosity.DETAILED:
            return self._format_error_detailed_plain()
        elif verbosity == Verbosity.COMPLETE:
            return self._format_error_complete_plain()
        elif verbosity == Verbosity.DEBUG:
            return self._format_error_debug_plain()
        else:
            return ""
    
    def _format_error_summary_plain(self) -> str:
        """Format basic error info for SUMMARY level."""
        from datetime import datetime
        
        lines = []
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Header
        lines.append(f"[{timestamp}] ERROR")
        
        # Show error info
        if hasattr(self, 'info'):
            lines.append(f"Error: {self.info.code}")
            lines.append(f"{self.info.message}")
            lines.append(f"Category: {self.info.category}")
        else:
            lines.append("Unknown error")
        
        return "\n".join(lines)
    
    def _format_error_detailed_plain(self) -> str:
        """Format detailed error for DETAILED level."""
        # TODO: Implement - show error with context
        return "ERROR DETAILED - TODO: Implement"
    
    def _format_error_complete_plain(self) -> str:
        """Format complete error for COMPLETE level."""
        # TODO: Implement - show full error
        return "ERROR COMPLETE - TODO: Implement"
    
    def _format_error_debug_plain(self) -> str:
        """Format full JSON debug for DEBUG level."""
        # TODO: Implement - show error as formatted JSON
        return "DEBUG ERROR - TODO: Implement"
