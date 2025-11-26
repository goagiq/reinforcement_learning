"""
Console color utilities for better visibility of errors and warnings.
"""

# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

def red(text: str) -> str:
    """Wrap text in red color"""
    return f"{RED}{text}{RESET}"

def green(text: str) -> str:
    """Wrap text in green color"""
    return f"{GREEN}{text}{RESET}"

def yellow(text: str) -> str:
    """Wrap text in yellow color"""
    return f"{YELLOW}{text}{RESET}"

def blue(text: str) -> str:
    """Wrap text in blue color"""
    return f"{BLUE}{text}{RESET}"

def bold(text: str) -> str:
    """Wrap text in bold"""
    return f"{BOLD}{text}{RESET}"

def error(text: str) -> str:
    """Format error message in red and bold"""
    return f"{RED}{BOLD}{text}{RESET}"

def warn(text: str) -> str:
    """Format warning message in yellow"""
    return f"{YELLOW}{text}{RESET}"

def success(text: str) -> str:
    """Format success message in green"""
    return f"{GREEN}{text}{RESET}"

def info(text: str) -> str:
    """Format info message in blue"""
    return f"{BLUE}{text}{RESET}"

