import logging

class LoggerFactory:
    COLORS = {
        'INFO': '\033[94m',     # Blue
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'DEBUG': '\033[90m',    # Grey
        'CRITICAL': '\033[95m', # Magenta
    }
    RESET = '\033[0m'

    class ColorFormatter(logging.Formatter):
        def format(self, record):
            color = LoggerFactory.COLORS.get(record.levelname, LoggerFactory.RESET)
            message = super().format(record)
            return f"{color}{message}{LoggerFactory.RESET}"

    @staticmethod
    def get_logger(name: str, level=logging.INFO) -> logging.Logger:
        formatter = LoggerFactory.ColorFormatter("%(asctime)s [%(levelname)s] %(message)s")

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Avoid duplicate handlers when re-importing
        if not logger.handlers:
            logger.addHandler(handler)

        # ✨ Fix: disable propagation to root logger
        logger.propagate = False

        return logger