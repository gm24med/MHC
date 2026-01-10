import logging


def get_logger(name: str = "mhc") -> logging.Logger:
    """Return a module-level logger for mhc."""
    return logging.getLogger(name)
