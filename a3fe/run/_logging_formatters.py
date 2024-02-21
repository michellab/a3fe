"""Custom logging formatters for the a3fe package."""

import logging as _logging


class _A3feFormatter(_logging.Formatter):
    """Formatter for the simulation runner logger."""

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
