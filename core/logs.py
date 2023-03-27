"""Class to setup Logging for project."""
import logging
import os
import sys
from datetime import datetime
from logging import FileHandler, StreamHandler
from pathlib import Path
from typing import Dict, List


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None


class ProjectLogger:
    """Return if the debugger is currently active"""

    log_format = f"[%(asctime)s] " \
                 f"[{Path().cwd().parts[-1]}] " \
                 f"[%(name)s] " \
                 f"[%(process)d] " \
                 f"[%(levelname)s] " \
                 f"%(message)s"

    project_name = Path().cwd().parts[-1]

    def __init__(self, logger_name: str, log_dir: str = "logs") -> None:

        self.logger = logging.getLogger(logger_name)
        self.active_debugger = debugger_is_active()
        self.now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = log_dir

        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(ProjectLogger.log_format)

        stream_handler = StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        if not self.active_debugger:
            if not os.path.exists(self.log_dir):
                self.d(f"{self.log_dir} does not exist, creating it.")
                os.makedirs(self.log_dir)
            log_filename = f"{ProjectLogger.project_name}_{self.now}.log"
            self.log_file = Path(log_dir, log_filename).absolute()

            file_handler = FileHandler(self.log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def d(self, message: str) -> None:
        """Debug log."""
        self.logger.debug(message)

    def i(self, message: str) -> None:
        """Info log."""
        self.logger.info(message)

    def w(self, message: str) -> None:
        """Warning log."""
        self.logger.warning(message)

    def e(self, message: str) -> None:
        """Error log."""
        self.logger.error(message)

    def c(self, message: str) -> None:
        """Critical log."""
        self.logger.critical(message)

    def setup_info(self, kwargs: Dict, exclude_list: List = []) -> None:
        """Log setup information."""
        if self.active_debugger:
            self.i(f"Initialising logger in debug mode.")
        else:
            self.i(f"Initialising logger and writing to file: {self.log_file}")
        for number, handler in enumerate(self.logger.handlers):
            self.i(f"Handler {number}: {handler}")
        self.i(f"Executing with arguments:")
        for k, v in kwargs.items():
            if k not in exclude_list:
                self.i(f"--- Argument: {k} - {v}")
