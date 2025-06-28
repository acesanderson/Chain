from enum import Enum

class Verbosity(Enum):
    """
    SILENT - Obviously nothing shown
    PROGRESS - Just the spinner/completion (your current default)
    SUMMARY - Basic request/response info (level 2 in your spec)
    DETAILED - Truncated messages in panels (level 3)
    COMPLETE - Full messages in panels (level 4)
    DEBUG - Full JSON with syntax highlighting (level 5)
    """

    SILENT = ""
    PROGRESS = "v"
    SUMMARY = "vv"
    DETAILED = "vvv"
    COMPLETE = "vvvv"
    DEBUG = "vvvvv"
