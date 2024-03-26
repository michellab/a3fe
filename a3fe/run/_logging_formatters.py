"""Custom logging formatters for the a3fe package."""

import logging as _logging


class _A3feStreamFormatter(_logging.Formatter):
    """Stream formatter for the simulation runner logger."""

    # From https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    blue = "\x1b[34;20m"
    cyan = "\x1b[36;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )
    format_str = "%(levelname)s - %(asctime)s - %(name)s - %(message)s"

    FORMATS = {
        _logging.DEBUG: grey + format_str + reset,
        _logging.INFO: blue + format_str + reset,
        _logging.WARNING: yellow + format_str + reset,
        _logging.ERROR: red + format_str + reset,
        _logging.CRITICAL: bold_red + format_str + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = _logging.Formatter(log_fmt)
        return formatter.format(record)


class _A3feFileFormatter(_logging.Formatter):
    """Stream formatter for the simulation runner logger."""

    format_str = "%(levelname)s - %(asctime)s - %(name)s - %(message)s"

    FORMATS = {
        _logging.DEBUG: format_str,
        _logging.INFO: format_str,
        _logging.WARNING: format_str,
        _logging.ERROR: format_str,
        _logging.CRITICAL: format_str,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = _logging.Formatter(log_fmt)
        return formatter.format(record)
