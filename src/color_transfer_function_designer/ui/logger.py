import logging
import pathlib
from datetime import datetime

LOG_DIR = pathlib.Path("logs")
if not LOG_DIR.exists():
    LOG_DIR.mkdir()

LOG_FILE_PATH = LOG_DIR / pathlib.Path(
    datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")[:-3] + ".log"
)


def install_handlers(logger: logging.Logger):
    # 2. Create handlers (Console and File)
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(str(LOG_FILE_PATH))
    c_handler.setLevel(logging.WARNING)
    f_handler.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)

    # 3. Create formatters and add them to handlers
    c_format = logging.Formatter(
        "{asctime} - {filename}:{lineno} - {levelname} - {message}", style="{"
    )
    f_format = logging.Formatter(
        "{asctime} - {filename}:{lineno} - {levelname} - {message}", style="{"
    )
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # 4. Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
